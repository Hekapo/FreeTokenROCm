"""Managed 64K entry point; no profiling or per-request patches."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

destination = os.environ.get('FT_TRITON_STABLE_KNOB_OUTPUT')
if destination:
    # PyTorch preloads the matching ROCm SDK DLLs before libtriton is linked.
    # Importing Triton first can bind it to the system HIP DLL on Windows.
    import torch
    import triton
    from stable_triton_knob import pin_stable_amd_knob
    initialized = pin_stable_amd_knob(triton)
    initialized.update(pid=os.getpid(),at=datetime.now(timezone.utc).isoformat(),module=__name__,
                       torch_version=torch.__version__,torch_imported_first=True,
                       finite_diagnostics_dir=os.environ.get('FT_FINITE_DIAGNOSTICS_DIR'),
                       finite_start_forward=os.environ.get('FT_FINITE_START_FORWARD','1'))
    with Path(destination).open('a',encoding='utf-8') as sink:
        sink.write(json.dumps(initialized)+'\n')

if __name__=='__main__':
    if not destination:
        raise RuntimeError('Managed stable knob output is required')
    from freetoken.cli import main
    sys.exit(main())
