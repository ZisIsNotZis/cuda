# cuda 已确立的事实与结论

本文件只搬运本仓库内已有的事实，不新增实验、不新增数字、不做质量评价。每条给出出处（文件:行）。

状态定义：**成立** = 仓内代码/文档直接支撑；**部分** = 仓内只有文字声明，没有对应的可执行验证；**已撤回** = 仓内文件显示该结论已被回退或不再成立。

先说清一件事：**本仓库不含任何测量数据**。所有性能数字都在仓外（见负结果 N1），因此本文件里没有任何速度、利用率、加速比。

## 已确立的结论

| # | 结论 | 关键数字 | 出处 | 状态 |
|---|---|---|---|---|
| 1 | 仓里有三条并存的 tensor-core 路径：`wmma` 片段 API、`mma.sync` 内联汇编、以及 `cp.async` 异步拷贝流水线 | 3 条 | `mm.cu:62-64`（wmma）、`test.cu:457`（mma.sync）、`test.cu:38`（cp.async） | 成立 |
| 2 | `mma.sync` 用的固定形状是 m16n8k16、f16 输入 f16 累加，在多个文件里反复出现 | 1 个形状 | `test.cu:457-459`、`test.cu:558`、`test2.cu:171-172` | 成立 |
| 3 | `cp.async` 只在 `test.cu` 与 `testbak.cu` 中以 16 字节粒度搬运 | 16 B | `test.cu:38`、`testbak.cu:136`、`testbak.cu:140` | 成立 |
| 4 | 融合注意力有 4 个版本的 kernel 名：`atn_ker`（两版）、`satn_ker`、`atn2_ker`；当前生效的是 `atn2_ker` | 4 个 | `mm.cu:191`、`mm.cu:279`、`mm.cu:334`、`mm.cu:380`、`mm.cu:445` | 成立 |
| 5 | `mm.cu` 的 GEMM 用 `cooperative_groups` 的 `memcpy_async`，block 内按 `threadIdx.x`/`threadIdx.y` 二维分工，每个 fragment 单独 `load_matrix_sync` | — | `mm.cu:4-5`、`mm.cu:59-75` | 成立 |
| 6 | warp 级排序有两个 kernel（`sort84_ker`、`sort4_ker` 的 8 位与 16 位重载），都用 `__maxnreg__(40)` 压寄存器 | 40 个寄存器上限 | `test4.cu:17`、`test4.cu:62`、`test4.cu:111` | 成立 |
| 7 | `test4.cu` 另有一个 warp 级归约 kernel，输入按 `int4` 向量化 | — | `test4.cu:159` | 成立 |
| 8 | `test7` 的 harness 用单个模板 kernel 覆盖全部配置：`m_ker` 带 18 个模板参数，`__maxnreg__(255)` | 18 个模板参数 / 255 | `test7/m.h:98-99` | 成立 |
| 9 | `test7` 的扫描规模是 (N+1)³ 个变体：`Makefile` 三重循环生成声明与调用，`N` 由命令行给 | (N+1)³ | `test7/Makefile:9-10`、`test7/Makefile:19-20`、`test7/Makefile:18` | 成立 |
| 10 | 该公式的落盘证据：`test7/1/m.cu`（N=1）恰好 8 个 kernel 声明与 8 次调用 | 8 = 2³ | `test7/1/m.cu:1-8`、`test7/1/m.cu:9-18` | 成立 |
| 11 | `test5` 的扫描不同：按 2 的幂递增 tile 参数（`x*=2` 三重循环），并对每个组合生成一个 `.cu` | 步长 ×2 | `test5/Makefile:15`、`test5/Makefile:17` | 成立 |
| 12 | 剖面只采一个指标 `gpu__time_duration.sum`，CSV 由 `ncu` 二次导出 | 1 个指标 | `test7/Makefile:3-7`、`test6/Makefile:3-4` | 成立 |
| 13 | `testbak.cu` 的 GEMM 用 `__launch_bounds__(BND)` 且模板默认值写死：BND=256、TX=TY=X=Y=4、Z=1、MOD=4 | 256 / 4 | `testbak.cu:54` | 成立 |
| 14 | `test3.cu` 是最小复现：`barrier.cta.sync` 内联汇编 + block `dim3(2,2,32)`，只打印 `cudaGetErrorString` | 8 行 | `test3.cu:1-9` | 成立 |
| 15 | 许可证为 AGPL-3.0-only | 1 个许可证 | `README.md:31`、`LICENSE` | 成立 |

## 负结果

| # | 结论 | 关键数字 | 出处 | 状态 |
|---|---|---|---|---|
| N1 | 仓库不含任何测量数据：`ncu` 的 CSV/`.ncu-rep` 未提交，只提交了解析它们的 notebook | 0 个数据文件 | `README.md:20-21` | 成立 |
| N2 | 没有统一入口，也没有 pass/fail 冒烟测试；每个文件自包含、单独 `nvcc` 编译 | — | `README.md:13-16` | 成立 |
| N3 | 实验 kernel 的运行本身就是结论未知的：可以报 CUDA 错误 | — | `README.md:17`、`test3.cu:8`（只打印错误串） | 成立 |
| N4 | 不是库，没有消费者 | 0 个消费者 | `README.md:10-11` | 成立 |
| N5 | 未使用 Hopper/Blackwell 的 `wgmma`/`tcgen05`；`wmma` 属 Volta/Turing 一代 API，与新写的 `mma.sync` 并列存在 | 2 代 API | `README.md:25-27` | 成立 |
| N6 | 迁移时凭据/数据集被丢弃、未带入，且原目录未被修改——即本仓不是完整快照 | — | `docs/project-status.md:15-16`、`docs/project-status.md:21-22` | 成立 |

## 撤回 / 回退记录

局部回退，仓内有痕迹但**没有写明原因**：

| # | 记录 | 关键数字 | 出处 | 状态 |
|---|---|---|---|---|
| R1 | 早期 `atn_ker` 实现整段被注释，不参与编译 | 1 段 | `testbak.cu:1` | 已撤回 |
| R2 | `mm.cu` 里 `satn_ker` 整段被注释 | 1 段 | `mm.cu:191` | 已撤回 |
| R3 | `mm.cu` 里旧版 `atn_ker`（带 `NKV` 参数那版）整段被注释 | 1 段 | `mm.cu:279` | 已撤回 |
| R4 | `mm.cu` 的 `main` 中 5 处 `atn_ker` 启动语句被注释，只剩 `atn2_ker` 一处生效 | 5 处 | `mm.cu:445` 对照 `mm.cu:451-475` | 已撤回 |

除此之外没有结论级撤回记录：本仓从未发布过带数字的结论，也就没有可撤回的数字。

## 未知（仓内无法确定，不做推断）

- 任何性能数字：单次 kernel 时延、加速比、SM 利用率、达成峰值百分比——仓内 0 个数据文件（N1），全部未知。
- 各 kernel 的数值正确性：没有测试、没有对拍、没有校验和，未知。
- 上述 5 处注释的真实原因（是回归、是慢、还是只是整理），仓内无记录，未知。
- 这些代码在当前机器/驱动上能否编译运行：本任务禁止运行，未复跑，未知。
- 迁移时被丢弃的文件清单：`docs/project-status.md:15-16` 只给了类别，没有文件名，未知。
