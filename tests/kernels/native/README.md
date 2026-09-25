# Native metadata regression definitions

Status: WRITTEN, NOT_COMPILED, NOT_RUN. No new execution is authorized by this file.

`test_symbolic_size.cpp` is a standalone C++20 regression source, not a pytest module.
It includes the actual `freetoken/tensor.h` and exercises SymbolicSize, SizeRef indirectly,
and TensorMatcher. It does not allocate GPU memory, create a GPU context, launch kernels,
import Python packages, or load/JIT FreeToken. A fake ROCm device tag appears only in a
metadata-rejection case and must never be passed to a GPU consumer.

There are 26 named scenarios. They are definitions, not passed tests. No build target,
CI workflow, compiler invocation, or auto-discovery bridge has been added. Compiling even
this host-only source remains prohibited until separate explicit authorization.

A future native build needs C++20 and a coherent set of FreeToken, TVM-FFI and DLPack
headers (and any link inputs the actual toolchain requires). Do not infer those paths
from a `.venv` directory or use this source as a reason to install packages. Header/API
references do not establish identity with the user's installed post3 distribution.

Fixtures own shape/stride arrays and storage through each complete matcher expression.
Do not keep a TensorMatcher or references to its initializer lists beyond that expression.
Expected values are direct metadata contracts, not a reimplementation of SymbolicSize.
The assertions remain active with NDEBUG because they do not use the C assert macro.

Not covered: GPU empty-input launch policy, count values on device, payload copies,
TVM-FFI/PyTorch stream agreement, host mapping, or compiled-cache freshness. In particular,
`with_dtype<T>(symbol)` / `with_device<D>(symbol)` have a separate open options-forwarding
issue; this source tests equality across symbols and no-symbol allowlists, not that issue.
