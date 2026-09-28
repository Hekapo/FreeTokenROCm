# Experimental Windows ROCm 10 source build

This is a recorded profile for an RX 9070 XT (`gfx1201`) and Python 3.12 on
Windows. It is not a release wheel or a claim that every Windows configuration
works. The local ROCm 10 environment passed GPU sanity and targeted tests; a
fresh install using the commands below has not been run end to end.

## Compatibility first

The [ROCm 10 compatibility matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html)
lists Windows 11 **25H2**, Adrenalin **26.8.1**, and `gfx1201` for the RX 9000
series. Check the GPU, Windows release, and driver before installing. Our local
validation host reported Windows 11 **24H2** (build 26100), so its successful
tests are evidence for that machine, not confirmation of official OS support.

Install Visual Studio 2022 Build Tools with the C++ workload and a Windows SDK.
Run the source build from a Visual Studio Developer PowerShell, so `cl.exe`,
`link.exe`, headers, and import libraries are available.

## Create an isolated environment

Use `pip` for this profile. The `pyproject.toml` uv source pins select CUDA
indexes, which are inappropriate for the Windows ROCm build.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel ninja
python -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ "rocm[libraries,devel,device-gfx1201]==10.0.0" "torch[device-gfx1201]==2.13.0+rocm10.0.0"
rocm-sdk init
python -m pip install "triton-windows==3.8.0.post28"
```

The ROCm SDK and PyTorch package selection follows AMD's
[ROCm installation guide](https://rocm.docs.amd.com/en/latest/install/rocm.html)
and [PyTorch guide](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html).
FreeToken does not require `torchvision` or `torchaudio`; they were absent in
the validated environment.

Keep `triton-windows` at **3.8.0.post28** for this profile. The initial ROCm 10
candidate produced incorrect MoE alignment results and a HIP launch failure
with Triton 3.8 on `gfx1201`. The source fixes in this fork route the affected
alignment and QSA kernels around those failures; targeted tests and a 30-minute
Qwen3.6 service soak passed with this exact Triton version in local candidate
stages. The merged `main` commit has passed focused tests and a launch preflight;
it has not had another 30-minute soak. The validated Windows
`cp312` wheel has SHA-256
`06435b922ebabbfcb3fb0f4b4b2257a2676dd39cc263a562be356bd58c15c9b8`.
The separate 3.7.1.post27 environment remains available for local rollback.

## Prepare the source build

Use a checkout of this fork containing the Windows ROCm source fixes and
Triton 3.8 workarounds. The upstream Linux/CUDA checkout and commit
`296a80e5` alone do not include the validated source. From the prepared
checkout and the activated environment:

```powershell
$SdkDevel = python -c "import importlib.util as u; print(next(iter(u.find_spec('_rocm_sdk_devel').submodule_search_locations)))"
$env:ROCM_HOME = $SdkDevel
$env:ROCM_PATH = $SdkDevel
$env:HIP_PATH = $SdkDevel
$env:HIP_PLATFORM = 'amd'
$env:PYTORCH_ROCM_ARCH = 'gfx1201'
$env:FREETOKEN_ROCM_ARCH = 'gfx1201'
$env:TVM_FFI_ROCM_ARCH_LIST = 'gfx1201'
$env:TRITON_LIBHIP_PATH = Join-Path $SdkDevel 'bin\amdhip64_7.dll'
$env:CC = Join-Path $SdkDevel 'lib\llvm\bin\clang-cl.exe'
$env:DISTUTILS_USE_SDK = '1'
$env:MAX_JOBS = '1'
$env:PATH = "$SdkDevel\bin;$SdkDevel\lib\llvm\bin;$env:PATH"
python -m pip install --no-build-isolation -e .
python -m pip check
```

`--no-build-isolation` makes the extension build use the already installed ROCm
PyTorch rather than resolving a different PyTorch wheel for the build. The
project's `torch>=2.11,<2.14` range admits the pinned 2.13 build. Do not
install the CUDA-only `freetoken[accel]` extra in this profile.

## Check the installed versions

```powershell
python -c "from importlib import metadata as m; names=['rocm','rocm-sdk-devel','rocm-sdk-device-gfx1201','torch','amd-torch-device-gfx1201','amd-torch-device-gfx12-0','triton-windows','apache-tvm-ffi']; print('\n'.join(f'{n}=={m.version(n)}' for n in names))"
python -c "import torch; print(torch.__version__, torch.version.hip)"
```

The recorded environment used Python 3.12.0, ROCm SDK 10.0.0, PyTorch
2.13.0+rocm10.0.0, `amd-torch-device-gfx1201` and its `gfx12-0` companion at
2.13.0+rocm10.0.0, `triton-windows` 3.8.0.post28, and
`apache-tvm-ffi` 0.1.13.post3. `pip check` reported no broken requirements.
The package checks above do not exercise GPU kernels, source extension builds,
model loading, or long-running stability.
