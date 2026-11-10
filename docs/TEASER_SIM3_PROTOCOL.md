# TEASER++ 未知尺度 Sim(3) 复现协议

## 范围与证据边界

本实验只比较 **TEASER++、Ours**，所有表格最后一行为 Ours。输入是 Stanford 扫描模型顶点构造的受控合成对应点，遵循 TEASER 论文未知尺度实验；不是原生跨传感器真实尺度数据。

主实验对应 **Bunny / Fig. 6(c)**。其他三模型按同一未知尺度协议运行，但标为扩展：Appendix S / Fig. 14 仅展示 TEASER 等方法的旋转和平移误差，未给出 TEASER++ 的四对象未知尺度箱线图，不能声称本次其他三对象结果直接复现了该图。

论文与已发布官方仓库没有给出 Fig. 6(c) 的完整生成/绘图脚本、原始随机试验数据、精确重建 PLY 文件名。以下未知细节显式固定为复现假设。**这是按可核实论文条件重建协议的实测结果，不能宣称与作者输入或数值逐位一致，也不能宣称所有未公开细节已严格复现。**

## 固定原始依据

- 论文：[arXiv:2001.07715v2](https://arxiv.org/abs/2001.07715v2)，2020-10-17。PDF 第一页印刷标识确认版本；44 页。
- PDF SHA256：`1730c0d52bf321def4f281f1250196f20500318360132349dade26c646fa9313`。
- [官方源码固定提交](https://github.com/MIT-SPARK/TEASER-plusplus/tree/52a9c52ee7d4c838c5e8a75458c33178be5bfb70)：`52a9c52ee7d4c838c5e8a75458c33178be5bfb70`。本地受版本控制源码无修改；安装分发包版本 1.1.0。
- 已核查官方 master 与 develop 文件树、`test/benchmark/registration-benchmark.cc`、六个静态 benchmark fixture、`matlab/` 和 `examples/`。没有发现 Fig. 6(c) 批量生成及绘图脚本。静态 benchmark 的重复计时次数不是论文 Monte Carlo 次数。
- [Stanford 官方模型及使用说明](https://graphics.stanford.edu/data/3Dscanrep/) 保存为数据目录的 `stanford/SOURCE_AND_USAGE.html`；保留压缩包内 README。
- 论文、文本、官方分支与文件树快照保存于 `/path/to/data/sim3_benchmarks/papers/`。

## 逐项协议

| 项目 | 本次固定值 | 依据 / 状态 |
|---|---|---|
| 对应关系 | `dst = s R src + t + noise`，目标对应点替换生成外点 | 论文 eq. (4)、§XI-A |
| 点数 | 每次 N=100 | §XI-B；不使用模块实验 N=40 |
| 外点比例 | 60%、70%、80%、90% | 直接检查 PDF 第 15 页 Fig. 6(c) 四个横轴位置 |
| 次数 | 每模型每比例 40 次；Bunny 160 组输入 | §XI-A 通用统计次数，§XI-B 沿用 |
| 原模型归一化 | 点云放入 `[0,1]^3` | §XI-A |
| 尺度 | `1 ≤ s ≤ 5`，需要估计 | §XI-A、Fig. 6(c) |
| 旋转、平移 | `R ∈ SO(3)`、`||t|| ≤ 1` | §XI-A |
| 噪声 | 各点独立 `N(0, 0.01² I)`，整向量范数超过 β 则重新采样 | §XI-A |
| 噪声界 | `β=5.54σ=0.0554`；`cbar²=1` | §XI-A 及实验 Implementation Details |
| 外点 | 替换目标点，均匀采样于半径 5 球体内部 | §XI-A；不是球面、不是目标点加大噪声 |
| 尺度误差 | `abs(s_pred-s_gt)` | §XI-A；官方 benchmark 第 218 行 |
| 旋转误差 | `acos(clip((trace(R_predᵀ R_gt)-1)/2,-1,1))`，转为度 | §XI-A、Fig. 6(c) 的 deg 纵轴 |
| 平移误差 | `||t_pred-t_gt||₂` | §XI-A；图纵轴标 m，本次注明归一化坐标单位 |
| 统计展示 | 每条件误差与时间箱线图 | Fig. 6(c)；不按精度筛选样本 |

不引入 3DMatch 尺度增强、log-scale error 或联合 RR 成功门限。不纳入 Fig. 6(a)/(b) 的已知尺度结果，不使用 N=1000、95–99% 外点结论解释本次未知尺度性能。

### 未公开细节的固定假设

1. PLY：各已下载压缩包内最高分辨率重建。Bunny 使用 `bun_zipper.ply`，Armadillo 使用 `Armadillo.ply`，Dragon/Buddha 使用 `*_vrip_res2.ply`。官方示例中另有 `bun_zipper_res3.ply`，但示例并非 Fig. 6(c) 的生成脚本，无法据此确定作者实验重建分辨率。
2. 每次从顶点均匀无放回抽取 100 点，然后减各轴最小值、三个轴共同除以最大轴跨度，保持几何形状并放入单位立方体。每次重采样；作者没有说明是否固定同一 100 点集合。
3. s 连续均匀 `[1,5]`；R 使用四维独立高斯归一化四元数的 Haar 分布；t 使用原点中心单位球内体积均匀分布。原文只给出范围。
4. 外点球以原点为中心；方向由归一化三维高斯生成，半径为 `5*u^(1/3)`。替换索引均匀无放回。不会用 GT 再排除偶然贴近模型的外点。
5. 随机流固定 NumPy PCG64 / SeedSequence：`[split_seed, object_index, outlier_percent, repeat]`。测试与验证种子完全分离。原作者 RNG、种子、各条件的随机数复用方式未公开。
6. 相同输入保存在 float32；TEASER++ 输入为这些值无损提升的 double，Ours 输入为 float32。源点转 float32 后才生成目标，避免源舍入改变噪声模型；目标舍入误差上界在正确性测试中单独容差检查。两个方法看到完全相同的坐标和顺序，内部算术精度按各自原生实现。
7. 使用 Q1/median/Q3、1.5 IQR 须线和所有离群点。原作者的精确分位数插值和画图参数未公开，本次采用 NumPy 线性百分位数与 Matplotlib 默认须线实现。

以上均在 `configs/teaser_sim3_protocol.json` 中固定，不冒充作者代码中的定义。

## 数据完整性

`scripts/prepare_teaser_stanford.py` 完整读取 gzip 至 EOF 校验 CRC，对 tar 检查绝对路径、父目录逃逸和链接；安全解压、重复执行不覆盖修改过的文件。

`stanford/manifest.json` 记录原始 URL、字节数、压缩包及选定 PLY 的 SHA256。这些是下载后的本地固定指纹，**不是 Stanford 发布的校验和**。已下载四包均通过检查，无需重复下载。生成试验只保存 N=100 的小型 NPZ，包含源、目标、GT、顶点索引、归一化参数、噪声、外点索引及随机种子；不复制整套模型。

## 求解器与计时

TEASER++：`estimate_scaling=True`，`noise_bound=0.0554`，`cbar2=1`；GNC_TLS、100 次、factor=1.4、cost threshold=1e-12 来自官方 `test/benchmark/registration-benchmark.cc:185`。CHAIN、精确最大团来自固定源码默认参数，`kcore_heuristic_threshold=1` 明确禁用提前跳过精确最大团的启发式。最大团时间上限保持 3600 秒；不是 RANSAC 的 60 秒预算。

使用同一官方提交另建 Release（`-O3 -DNDEBUG`）、`ENABLE_DIAGNOSTIC_PRINT=OFF`，不修改算法源码。这样逐步诊断日志不进入求解计时。系统原有 TEASER++ 安装保留。运行快照保存加载的二进制哈希、CMake 配置和编译命令。

Ours：核心仍为已有 C++/LibTorch Sim(3)，没有调整算法。独立 Bunny 验证种子，每比例 4 次共 16 组输入。预设网格 `epsilon_scale ∈ {0.1,0.2}`、`k_neighbors ∈ {20,80}`，共同 `max_points=100`、`inlier_threshold=β`，其他默认参数。以验证集平均旋转误差选择，异常在**参数选择阶段**赋 180°，同分选较早候选。这不是测试 RR 定义。选定值保存为 `ours_frozen.json` 后全测试集不再调参。

两方法同机 CPU，OMP/MKL/OpenBLAS 一线程、Torch intra-op 一线程/inter-op 一线程。保存实际 CPU 型号、进程可用 CPU 列表、Torch/NumPy/Python 版本；共享主机不是独占节点。CUDA 不可用时不报告 GPU 耗时。

每次计时：输入数组转换、设备传输和参数对象准备在外；TEASER 从构造求解器到 `solve()` 返回，Ours 为 `register_sim3()` API 调用。包括绑定开销和 C++ 内部分配，不包括 GT 指标、NPZ I/O、生成时间及 JSON 日志。参数对象构造略不同于官方 C++ benchmark 计时起点，因此仅比较本机统一边界的两方法，不逐值比较论文机器时间。GPU 如另行扩展需同步计时并独立报告。启动后先用独立验证种子做两方法不计时预热；按 repeat 奇偶交换先后次序。

## 失败、统计与续跑

- 有限但不准确的预测完整进入误差箱线图，不能按 GT 门限删除。
- 异常、非有限/非 Sim(3) 矩阵、TEASER 最大团过小导致提前退出计为 `failed`；误差存 null，保留异常或有限的无效预测。没有单位阵回退。
- 对失败无法定义有限误差，图另给失败计数面板；统计表显式列失败/观测数量。有限样本分位数列带 `_finite`；`mean_all` 对任一失败返回 `+inf`，缺失试验返回 `incomplete`。不会把仅正常返回样本均值冒充全部试验均值。
- 原图没有官方 Sim(3) RR 表。本次 Markdown/CSV 是同批箱线图数据的补充汇总，每个条件 TEASER++ 一行、Ours 最后一行。
- 每次求解 fsync JSONL，以 `(trial_id, method)` 唯一去重。输出目录文件锁防止并发写。中断只修复最后一个未完成行；完整坏行或重复键报错。
- 续跑校验协议、输入 PLY、算法二进制、执行脚本、环境、参数哈希；再次生成随机输入并逐数组比对保存的 NPZ。配置改变必须新建输出目录。
- `summary.json.complete` 仅在全体预期 1,280 次求解键齐全时为 true；Bunny 主实验 320 次，三对象扩展 960 次。部分结果明确标 incomplete。

## 运行入口

Python 固定 `python`，在项目根运行：

```bash
bash registration/scripts/build_teaser_paper.sh
python -m unittest discover -s tests -p test_teaser_protocol.py -v
python registration/scripts/teaser_sim3_benchmark.py validate --output results/teaser_sim3_validation_release
python registration/scripts/teaser_sim3_benchmark.py run --output results/teaser_sim3_paper --ours-options results/teaser_sim3_validation_release/ours_frozen.json
python registration/scripts/teaser_sim3_benchmark.py summarize --output results/teaser_sim3_paper
```

重复 `run` 同一命令即续跑。`--limit-calls` 仅用于独立 smoke 目录，不把重复次数改为 smoke 数量。结果输出 `results.jsonl`、`trials/*.npz`、`run.json`、`provenance/`、`summary.csv/json`、`RESULTS.md`、四对象 PNG/PDF 箱线图。

数值结果见论文 Table/Appendix 与 `registration/README.md` 的结果摘要；本仓库不含 947M 原始运行归档。
