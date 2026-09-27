<!-- psplat-docs: audience=td -->
# psplat on Windows

psplat runs natively on Windows 11 in PowerShell. It needs no WSL, no
administrator rights, no compiler and no cloud account. Every command here
is also run literally by the Windows job in CI, so if this page is wrong the
build is red.

## 1. Install Python

Install Python 3.12 or 3.13 from python.org. In the installer, tick
**Add python.exe to PATH**. Then open a new PowerShell window and check:

```bash
python3 --version
```

```powershell
python --version
```

If `python` is not found but `py` is, the installer skipped the PATH step;
use `py -3.12` wherever this page says `python`.

## 2. Install psplat

psplat is not on PyPI yet. Until v0.1 is published, replace `psplat` in the
`pip install` line with the path to a wheel from a GitHub release.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install psplat
```

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
pip install psplat
```

`Set-ExecutionPolicy -Scope Process` lets this window run the activation
script. It needs no administrator rights and lasts only until the window
closes. In a new window, run it and the `Activate.ps1` line again.

The install is a pure-Python package plus PyTorch. On Windows, PyTorch from
PyPI is the CPU build, about 200 MB.

## 3. Check it works

```bash
psplat --version
psplat analyze scene.ply --cameras cams.json --device cpu
```

```powershell
psplat --version
psplat analyze scene.ply --cameras cams.json --device cpu
```

Paths may use either `\` or `/`. psplat handles both.

## Using an NVIDIA GPU

**Not yet available, and not yet verified on a Windows NVIDIA machine.**
The CUDA renderer is tracked in
[#23](https://github.com/zach/psplat/issues/23) and its Windows install in
[#29](https://github.com/zach/psplat/issues/29). This section describes
the intended path so the pieces are in one place; it will be tested before
it is relied on.

The plan is one extra, pinned to a Python, PyTorch and CUDA combination for
which gsplat publishes a prebuilt Windows wheel, so no compiler is needed:

<!-- doctest: skip -->
```bash
pip install "psplat[cuda]"
psplat analyze scene.ply --cameras cams.json --backend gsplat
```

```powershell
pip install "psplat[cuda]"
psplat analyze scene.ply --cameras cams.json --backend gsplat
```

Until then, `--backend auto` uses the portable PyTorch renderer, and says so.

### If no prebuilt wheel matches

gsplat then compiles its CUDA code the first time it runs. That needs two
installs, and both are large:

1. **Build Tools for Visual Studio 2022**, with the *Desktop development
   with C++* workload. This provides `cl.exe`.
2. **The CUDA Toolkit** whose version matches your PyTorch. Check which one
   with:

<!-- doctest: skip -->
```bash
python -c "import torch; print(torch.version.cuda)"
```

```powershell
python -c "import torch; print(torch.version.cuda)"
```

Run the first analysis from the **x64 Native Tools Command Prompt for VS
2022**, or from a PowerShell window where `cl` is on the PATH. The build
takes several minutes once, then is cached.

### If `nvcc` is not found

The CUDA Toolkit's `bin` folder is not on the PATH, or `CUDA_HOME` is
unset. Point both at the toolkit version that matches PyTorch, for example
12.8:

<!-- doctest: skip -->
```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
nvcc --version
```

```powershell
$env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
$env:PATH = "$env:CUDA_HOME\bin;$env:PATH"
nvcc --version
```

These settings last for the current window only.
