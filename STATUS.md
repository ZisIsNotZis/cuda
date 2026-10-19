# cuda 状态

一句话：手写 CUDA kernel 的个人实验场——`wmma`/`mma.sync` tensor-core GEMM、融合注意力、卷积转 GEMM、warp-shuffle 归约/排序、`cp.async`/cooperative-groups 流水线，外加模板元编程的网格搜索 harness 与 Nsight Compute 剖面（`README.md:3-6`）。

## 当前状态

**归档（2026-09-30 迁入，不再维护）。**

- 归档声明：`docs/project-status.md:5-6` —— 2026-09-30 从私有 `~/my` 集合迁出，为可查看而非任其腐坏；原目录未被修改。同一节 `docs/project-status.md:21-22` 写明"Not maintained. Kept for reference only."
- 迁移范围：`docs/project-status.md:15-16` 记录为逐字复制，除去垃圾二进制、缓存，以及任何触发凭据扫描器的文件；`docs/project-status.md:21-22` 记录迁移中发现的凭据/数据集被丢弃、未带入。
- 代码文字本身是 2024–2026 年间的当前写法（CUDA C++、`mma.h`/`wmma`、`cp.async`、`cooperative-groups`、`-arch native`、CUDA 12），`README.md:23-29` 的弃用检查结论是"nothing in the repo is built on a deprecated framework"，但语气按 `archive_only` 处理。
- 提交历史只有 3 个提交：2026-09-30 迁入（`1d5870f`）、2026-10-15 加许可证（`0758ec6`）、2026-10-15 仓库管家（`c374c19`）。代码文件自迁入后未再改动。
- 边界：本次只读仓内文件，未编译、未运行；`README.md:17` 明说这个实验 kernel 的运行可以报 CUDA 错误。

## 最新进展落在哪个文件

- 状态与边界：`docs/project-status.md`（分类/状态/证据/边界四节）。
- 是什么、怎么跑、弃用检查结论：`README.md`（尤其 `README.md:13-21` 的运行方式、`README.md:23-29` 的弃用检查）。
- 实现：根目录 `mm.cu`、`test.cu`、`test2.cu`、`test4.cu`、`testbak.cu`、`test3.cu`，以及 `test5/`、`test6/`、`test7/` 三套模板化网格搜索（`m.h` + `m.ipynb` + 按参数生成变体的 `Makefile`）。

## 下一步

仓内没有记录任何后续计划；`docs/project-status.md:21` 与 `README.md:28-29` 都把"不再维护、仅作参考"写成终态。若日后要复用，仓内唯一写明的入口是 `README.md:13-21`：每个文件自包含，无统一入口，按 `nvcc -arch native` 单文件编译；扫描变体在 `test6/`、`test7/` 下 `make`，`N=` 在命令行给（`README.md:18-20`）。
