# cuda — hand-written tensor-core & warp-level CUDA experiments

**What:** A personal lab of hand-written CUDA kernels exploring `wmma`/`mma.sync`
tensor-core GEMM, fused attention (QK^T·V), convolution-as-GEMM, warp-shuffle
reductions/sorts, and `cp.async` / cooperative-groups pipelines, plus a
template-metaprogramming grid-search harness profiled with Nsight Compute (`ncu`).

**Why it exists:** Notes-to-self from a deep dive into CUDA tensor-core
programming and the PTX/inline-asm that backs it. The point was to measure real
tile/thread-block configurations on hardware rather than trust docs. It is not a
library and has no consumers.

**How to run:** there is no single entrypoint. Each file is self-contained:
```
nvcc -arch native -o /tmp/t3 test3.cu && /tmp/t3
```
This is an experimental kernel, not a pass/fail smoke test; its run can report a CUDA error.
For the swept variants, `cd test6` or `cd test7` and `make` (needs `nvcc` and,
for the `.ncu-rep`/CSV targets, Nsight Compute `ncu`); tune `N=` on the make
command line. The notebooks (`test{5,6,7}/m.ipynb`) parse the `ncu` CSV dumps
for plotting; the CSVs themselves are not committed.

**Status:** deprecation check — this is a **current (2024–2026 era)** codebase:
CUDA C++, `mma.h`/`wmma`, `cp.async`, `cooperative_groups`, `-arch native`,
CUDA 12 toolchain. `wmma` is the older (Volta/Turing) tensor-core API alongside
newer `mma.sync`, and Hopper/Blackwell `wgmma`/`tcgen05` are not used here, but
nothing in the repo is built on a deprecated framework. It is an unpolished
scratch archive, not a maintained project — treat it as `archive_only` in tone
even though the code is publishable.

**License:** AGPL-3.0-only; see `LICENSE`.
