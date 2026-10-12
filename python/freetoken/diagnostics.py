"""Opt-in, synchronous first-nonfinite capture. Never use for speed measurements."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Mapping

import torch


MAX_SNAPSHOT_BYTES = 64 * 1024 * 1024


class FirstNonFiniteError(RuntimeError):
    pass


class FiniteDiagnostics:
    def __init__(self, directory: str | Path, *, start_forward: int = 1,
                 end_forward: int | None = None, snapshot_bytes: int = MAX_SNAPSHOT_BYTES):
        if start_forward < 1 or (end_forward is not None and end_forward < start_forward):
            raise ValueError("invalid finite diagnostic forward interval")
        if not 8192 <= snapshot_bytes <= MAX_SNAPSHOT_BYTES:
            raise ValueError("finite diagnostic snapshot must be between 8 KiB and 64 MiB")
        self.directory = Path(directory)
        self.start_forward, self.end_forward = start_forward, end_forward
        self.snapshot_bytes = snapshot_bytes
        self.forward = 0
        self.active = False
        self.metadata: dict = {}
        self.failure: FirstNonFiniteError | None = None

    def begin_forward(self, phase: str, requests: list[dict]) -> None:
        if self.failure is not None:
            raise self.failure
        self.forward += 1
        self.active = (phase == "prefill" and self.forward >= self.start_forward
                       and (self.end_forward is None or self.forward <= self.end_forward))
        self.metadata = {"forward": self.forward, "phase": phase, "requests": requests}

    def set_layer(self, layer: int) -> None:
        self.metadata["layer"] = layer
        self.metadata.pop("gdn", None)

    @staticmethod
    def describe(tensor: torch.Tensor) -> dict:
        return {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
                "stride": list(tensor.stride()), "device": str(tensor.device),
                "bytes": tensor.numel() * tensor.element_size()}

    @staticmethod
    def _copy(tensor: torch.Tensor, remaining: int) -> torch.Tensor | None:
        count = remaining // tensor.element_size()
        if count == 0:
            return None
        view = tensor.detach()
        # Narrow before copying, including noncontiguous views of large pools.
        # clone() prevents torch.save from serializing the backing storage.
        for dim in range(view.ndim):
            if view.numel() <= count:
                break
            slab = view.numel() // max(1, view.shape[dim])
            view = view.narrow(dim, 0, max(1, min(view.shape[dim], count // slab)))
        return view.to(device="cpu").clone(memory_format=torch.contiguous_format)

    def preserve(self, tensors: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Keep bounded pre-operation values when a kernel mutates its inputs."""
        result, remaining = {}, self.snapshot_bytes // 2
        for name, tensor in tensors.items():
            copy = self._copy(tensor, remaining)
            if copy is not None:
                result[name] = copy
                remaining -= copy.numel() * copy.element_size()
        return result

    @staticmethod
    def selected_states(source: torch.Tensor, indices: torch.Tensor) -> dict[str, torch.Tensor]:
        # Check only live indexed states, not uninitialized/unused pool slots.
        return {f"state_slot_{slot}": source[slot]
                for slot in dict.fromkeys(indices.detach().cpu().tolist())}

    def check(self, boundary: str, outputs: Mapping[str, torch.Tensor], *,
              inputs: Mapping[str, torch.Tensor] | None = None, metadata: dict | None = None) -> None:
        if not self.active:
            return
        if self.failure is not None:
            raise self.failure
        for name, tensor in outputs.items():
            if tensor.is_cuda and torch.cuda.is_current_stream_capturing():
                raise RuntimeError("finite diagnostics require cuda_graph_bs=0")
            finite = torch.isfinite(tensor)
            if not finite.all().item():
                flat = (~finite).reshape(-1).to(torch.uint8).argmax().item()
                index = []
                for size in reversed(tensor.shape):
                    index.append(flat % size)
                    flat //= size
                index.reverse()
                self._fail(boundary, name, index, outputs, inputs or {}, metadata or {})

    def _location(self, name: str, index: list[int], operation: dict) -> dict | None:
        gdn = self.metadata.get("gdn", {})
        requests = self.metadata.get("requests", [])
        lengths = [r["extend_len"] for r in requests]
        cu = gdn.get("cu_seqlens")
        if cu is not None:
            lengths = [b - a for a, b in zip(cu, cu[1:])]
        if name.startswith("state_slot_"):
            slot = int(name.removeprefix("state_slot_"))
            slots = gdn.get("indices", [])
            return {"state_slot": slot, "request": slots.index(slot) if slot in slots else None}
        axes = {"q": 1, "k": 1, "v": 1, "g": 1, "beta": 1, "A": 1,
                "w": 1, "u": 1, "v_new": 1, "o": 1, "hidden": 0, "out": 0,
                "mixed": 0, "core_out": 0, "conv_in": 0, "z": 0, "a": 0, "b": 0,
                "logits": 0, "routed": 0, "shared": 0, "gate": 0}
        if name == "h" and len(index) > 1:
            # h stores the state at each chunk's start, not its final state.
            position, chunks = index[1], True
        elif name in axes and len(index) > axes[name]:
            position, chunks = index[axes[name]], False
        else:
            return None
        bt = operation.get("BT", 64)
        for n, length in enumerate(lengths):
            size = (length + bt - 1) // bt if chunks else length
            if position < size:
                token = position * bt if chunks else position
                return {"request": n, "token_in_forward_sequence": token,
                        "chunk_in_forward_sequence": token // bt,
                        "absolute_token": requests[n]["cached_len"] + token if n < len(requests) else None,
                        "state_at_chunk_start": chunks}
            position -= size
        return None

    def _fail(self, boundary, name, index, outputs, inputs, metadata):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        target = self.directory / f"finite-{stamp}-{os.getpid()}"
        self.failure = FirstNonFiniteError(f"first nonfinite at {boundary}/{name}; capture: {target}")
        record = {"schema": 1, **self.metadata, "boundary": boundary, "tensor": name,
                  "first_bad_index": index, "operation": metadata,
                  "outputs": {k: self.describe(v) for k, v in outputs.items()},
                  "inputs": {k: self.describe(v) for k, v in inputs.items()},
                  "snapshot_limit_bytes": self.snapshot_bytes, "diagnostic_timing": True}
        record["location"] = self._location(name, index, metadata)
        try:
            target.mkdir(parents=True, exist_ok=False)
            # Reserve space for the archive header and metadata; actual size is
            # checked too. The first bad scalar survives even if a tensor is huge.
            metadata_budget = min(65536, self.snapshot_bytes // 4)
            remaining = self.snapshot_bytes - metadata_budget - min(1024 * 1024, self.snapshot_bytes // 4)
            snapshots = {"first_bad_value": outputs[name][tuple(index)].detach().cpu().clone()}
            remaining -= snapshots["first_bad_value"].element_size()
            saved = {}
            ordered = [(f"output/{name}", outputs[name])]
            ordered += [(f"input/{k}", v) for k, v in inputs.items()]
            ordered += [(f"output/{k}", v) for k, v in outputs.items() if k != name]
            for key, value in ordered:
                limit = remaining // 2 if key == f"output/{name}" else remaining
                copy = self._copy(value, limit)
                if copy is not None:
                    snapshots[key] = copy
                    remaining -= copy.numel() * copy.element_size()
                    saved[key] = {**self.describe(copy), "truncated": copy.shape != value.shape}
            archive = target / "tensors.pt"
            torch.save(snapshots, archive)
            record["saved"] = saved
            record["snapshot_file_bytes"] = archive.stat().st_size
            if record["snapshot_file_bytes"] > self.snapshot_bytes - metadata_budget:
                archive.unlink()
                record["snapshot_error"] = "archive overhead exceeded bounded snapshot budget"
                record["snapshot_file_bytes"] = 0
        except Exception as error:
            record["snapshot_error"] = repr(error)
        try:
            data = json.dumps(record, indent=2, ensure_ascii=True)
            if len(data.encode()) > min(65536, self.snapshot_bytes // 4):
                data = json.dumps({"boundary": boundary, "tensor": name,
                                   "first_bad_index": index, "snapshot_error": "metadata too large"})
            (target / "record.json").write_text(data, encoding="utf-8")
        except Exception as error:
            self.failure.args = (f"{self.failure}; could not write diagnostic metadata: {error!r}",)
        raise self.failure


def _from_env() -> FiniteDiagnostics | None:
    directory = os.environ.get("FT_FINITE_DIAGNOSTICS_DIR")
    if not directory:
        return None
    end = os.environ.get("FT_FINITE_END_FORWARD")
    return FiniteDiagnostics(directory, start_forward=int(os.getenv("FT_FINITE_START_FORWARD", "1")),
                             end_forward=int(end) if end else None)


# One process-local monitor; ordinary inference executes only a None/active branch.
FINITE_DIAGNOSTICS = _from_env()
