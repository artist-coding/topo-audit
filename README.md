# topo-audit

DeepMath 的 P1、P2 和正式划分统一先剥掉 boxed 包装并归一化，再剔除空答案、
Yes/No/True/False 和 converges/diverges（含 convergent/divergent）等已声明的判断标签。
这不能识别所有二选一题。下载可用 `python3 data/fetch_deepmath.py 10 --band 8.5`，
或对现有数据运行 `python3 -m data.loaders.deepmath --band 8.5`，检查过滤前后高频答案再划分。
数值答案如 0、1 不会因为出现频繁而被剔除；筛选版本为 `normalized_nonbinary_v2`。
下文旧 P1b 的准确率和工期来自过滤前题池，选主格前须重新测量并检查剩余题量。
P2 遇到缺失评估会停止并要求缓存续跑；其总费用包含图评估和去重后的单智能体调用。

把"学到的 MAS 拓扑相对等算力单智能体的收益"分解成四项并给出置信区间的实验框架。

论文设计见 [`topo-audit-docs/`](topo-audit-docs/)；已冻结的决定与硬规则见
[`CLAUDE.md`](CLAUDE.md)。本文件是给审阅者的入口。

## 要测的是什么

一条学习曲线走平了，有四种可能的原因，只有第一种是"框架到顶了"：

```
P(G_star)   ─── 空间里真正最优的拓扑          ← 天花板
P(G_hat_N)  ─── 按训练题挑出来的最优拓扑
P(G_learned)─── 优化器交出来的拓扑            ← 曲线停在这里
P(pi_single)─── 等 token 预算下最强的单智能体
```

四级阶梯，三个相邻间隔各对应一个动作：

| 间隔 | 等于 | 决策 |
| --- | --- | --- |
| P(G_star) − P(G_hat_N) | 估计项 E | 加任务 |
| P(G_hat_N) − P(G_learned) | **O + Δ_gen** | 继续搜 |
| P(G_star) − P(pi_single) | 容量 C | 扩空间 / 停 |

恒等式 `P(G_learned) − P(pi_single) = C − E − O − Δ_gen` 是伸缩和，无需假设。
`O + Δ_gen` 把训练目标上的量 O 换算成总体量，所以 Δ_gen 不是第四个待解释的东西，
而是汇率。详见 [`analysis/decompose.py`](analysis/decompose.py) 的模块文档。

## 审计空间 S₀

4 个节点固定顺序执行（Solver → Critic → Reviser → Finalizer），6 条可选前向边。
三个数都要在论文里报，因为它们各有用处：

| 数 | 含义 | 决定什么 |
| --- | --- | --- |
| **64** | 边掩码 | 优化器实际搜索的空间，决定搜索难度 |
| **34** | 带标号的可执行 DAG | 空间的名义大小 |
| **25** | 互不相同的计算 | 容量与估计项里的 log∣S∣；**张量按这个付钱** |

34 → 25 的塌缩来自"缺输入的 Critic 退化成 Solver 并用 Solver 提示"：
`n1→n3 n3→n4` 和 `n2→n3 n3→n4` 发出的是逐字节相同的请求。

闭式校验：`f(k) = Σ_m C(k−1,m)·Π_{j=1..m}(2^j−1)` 给出 34 / 422 / 11586（k=4/5/6）。

## 架构

```
数据 → 四角色划分 → 图空间 S₀ + π_single → 执行器（唯一端点出口）→ 缓存 → 分析
```

| 模块 | 职责 |
| --- | --- |
| [`space/`](space/) | Topology、剪枝、规范化、双身份 hash、S₀/S₁ 枚举 |
| [`data/`](data/) | MATH / DeepMath 加载、boxed 打分、固定种子四角色划分 |
| [`runtime/`](runtime/) | 端点客户端、执行器、SQLite 缓存、token 记账、π_single |
| [`optimizers/`](optimizers/) | 随机搜索、边级 REINFORCE、prune_from_dense、贪心局部搜索 |
| [`audit/`](audit/) | 建得分张量 |
| [`analysis/`](analysis/) | 四项分解、交叉拟合、bootstrap、平台期检验、报告 |

**两种图身份。** `graph_hash` 认带标号的图（优化器选的那个掩码），
`exec_hash` 认计算（缓存与张量按它寻址）。`space_id` 是 provenance 不是 identity，
所以一张张量同时服务 S₀ 和嵌套的 S₀-strict。

**张量一次建好，之后全是重放。** 25 行 ×（train 256×8 + audit 300×2 + val 300×2
+ test 1000）= 106,200 格 / 382,320 次节点调用。建好之后任意优化器、种子、预算、
N 值都是离线查表，零端点调用（`optimizers/sweep.py` 用 `ReplayOnlyEndpoint`，缺格直接报错）。
种子设为 20：重放虽然免费，但种子共享张量格子，50 个种子会让样本被复用 3.4 次；
而且收敛后各种子持有同一对图，种子对平台期检验的任务噪声没有帮助（见 CLAUDE.md）。

## 审阅时值得盯的几处

| 位置 | 为什么 |
| --- | --- |
| `analysis/decompose.py` 的 `check()` 文档 | 恒等式断言是**结构性成立**的（50 组随机数验过），只能抓公式回归，抓不住坏数据。真正的隐患是混用 P(G_star) 的两个估计量，由 `mixed_star_residual()` 量化 |
| `analysis/decompose.py` 的 plug-in vs 交叉拟合 | 两个估计量回答不同问题，故意分开：plug-in（audit 选图、test 报分）是 headline；交叉拟合估的是"选择流程"的期望，它与朴素 max 的差才是 winner's curse |
| `analysis/plateau.py` 的 frozen vs moved_flat | 把"incumbent 冻住"和"变了没涨"分开报。合并它们，就是把搜索卡住写成天花板 |
| `analysis/matrices.py` | 没跑到的格子保持 NaN。0 的含义是"答错了"，和"没跑"是两个断言 |
| `runtime/executor.py` 的 `render_inputs()` | 无入边节点不能拿到一个空的 "Received work:" 标题——否则投票图里的裸 solver 和 π_single 的 CoT 吃残缺提示，而 C 正是这两者的差 |
| `optimizers/base.py` 的预算定义 | 缓存命中和等价掩码仍然要花掉一次候选评估及其 token，否则重放过的搜索会获得免费算力 |
| `prompts/` | 四个角色模板 + selfcheck，E0 内不优化。改它们会触发 `bind_experiment` 报错（变了就是新的 exp） |
| `analysis/trajectory.py` | 各优化器每轮扣的评估数不同（1 / 8 / 5–7），先对齐到同一条预算轴再比较；按 cum_evals 精确查表曾让 reinforce_edge 从平台期检验里整个消失 |
| `runtime/single_agent.py` 的 `select_on_val_report_on_test` | π_single 在 val 上选、test 上只评估被选中的那一个，逐题分数让 C 成为配对差；截断规则与图一致 |
| `scripts/p2_spread.py` | 建张量前的门：看最好 − 第五名、交叉拟合的 C、top-5 配对差方差，而不只是最好 − 最差 |

## 已实测的端点约束

在 gateway（DeepSeek = ds-v4.1-flash）上测到的，都已写进代码：

- **必须绕过 clash 代理**，否则 502。而 HuggingFace 必须走代理——两边相反。
- **`n > 1` 被拒**，所以 k 个样本 = k 次调用。
- **思考默认开**，`thinking.type=disabled` 无效；P1c 发现 `reasoning_effort`（none / low /
  high / xhigh / max）能调。历史 E0 模板用默认，经 `model.request_options: {}` 显式记录；改了它就进
  `bind_experiment`，必须换 exp。
- **`max_tokens=1024` 会把实验做废**：思考链吃光额度，`content` 返回空，记 0 分。
  而 4 节点图有 4 次机会撞上这个上限，单节点只有 1 次——会人为造出"人多更差"。
  实测 8192 下截断率 3%，真端点冒烟 0%。
- **历史短提示吞吐约 1000–1200 生成 token/s。** 该次探测中并发 8 和 32 总量相同，
  64 开始排队，128 触发 429。这是短提示探测；长生成必须另测各并发的实际吞吐与失败耗时。

## 怎么跑

```bash
cp .env.example .env            # 填 gateway 的 key
python3 -m pip install -r requirements.txt
python3 -m pytest tests/ -q     # 默认跳过 slow；全量用 -m slow

python3 space/s0.py --nodes 4 --enumerate      # 应打印 34 / 25 / strict 16
python3 data/fetch_math.py                     # 需要代理；DeepMath 用 data/fetch_deepmath.py
python3 data/make_splits.py                    # MATH L4-5；DeepMath: --family deepmath --band 8.5

# 历史 E0 驱动顺序；先完成下文成本诊断/P2，并另建正式 DeepMath 配置，勿直接运行旧模板
# 当前 P2 冒烟命令见“现状”，运行条件尚未冻结
python3 audit/run_audit.py --config configs/e0.yaml --dry-run
python3 audit/run_audit.py --exp e0_smoke --splits val --limit-tasks 5   # 真端点冒烟
python3 audit/run_audit.py --config configs/e0.yaml        # 张量 + manifest_audit.json
python3 runtime/single_agent.py --config configs/e0.yaml   # π_single：val 上选，test 上报
python3 optimizers/sweep.py --config configs/e0.yaml       # 只重放
python3 analysis/report.py --config configs/e0.yaml        # report.json + decomposition_rows.jsonl
```

## 现状

issue #3 的长生成成本复核（2026-10-19，默认思考/stream/32k，并发 4）：
90 次节点调用，均值 **5,911** token，p50 **2,176**，p90 **26,724**，最大 **32,768**。
本轮新增 89 次调用（1 次命中此前诊断缓存），503,178 completion token / 1,524.78 秒
= **330 token/s**；无重试、无断流。角色统计及失败用量缺失标记保存在报告 `cost` 段。
按该吞吐和调用均值外推，P2（30,300 调用）约 **6.28 天**，候选张量
（571,320 调用）约 **118.45 天**，另加 val 单智能体前沿约 **6.16 天**。
这是 5 题、hand6 角色混合样本的粗估，不是工期保证，不能外推未实测的并发 8/16。
用户工期目标为正式张量 **1–2 周**（不含 P2）。下一步用独立配置
`configs/p2_dm80_90_v2_low_stream32k.yaml` 比较 low + 流式/32k，再对合适条件测并发 8/16。
保留 val=1,350、R=1/2 两个方案，按实测方差决定；全量 P2 尚未启动。

`cost` 统计每次物理尝试并保留未知用量；纯缓存重放时 fresh=0、吞吐和工期为 null。
`power.observed_repeats` 标记实测方差对应的 R；R≥2 时 `repeat_projection` 报题内/题间
分量及 R=1/2/4 的题量需求，假设图间及各次采样噪声独立，负的题间估计截为零并保留原值。
尚未实际运行两次采样时不推断这些分量，也不据此冻结 val 的题数与重复次数。

截断诊断保留原始极差，同时给出同题行标签置换的 95% 参考极差（同题的重复采样放在一起）。
至少 30 题时，仅在同时超过实际阈值和噪声参考时触发差异门槛；小冒烟显示 n/a。
这依赖零假设下行标签可交换，不拒绝零假设不等于证明截断差小于 3pp。
P2 另检查总截断率；张量门槛合并四个 split，各 split 仅作诊断。不得据小样本原始极差追着噪声加上限。


2026-10-19 已下载完整 10 个分片（103,022 题，源版本
`5cf055d1fe3d7a2eb19719ac020211469736ae44`）。v2 过滤及去重后，`[8.5,9.0)`
只有 1,423 题，无法容纳原定四角色划分；`[8.0,9.0)` 有 8,157 题。
本轮 P2 因此固定宽题池，计划 150 道留出题、每格两次采样。
验证集候选为 1,350 题 × 2，测试集保留 1,000 题；ε=2 分是待验证的精度目标。
正式 E0 划分和配置尚未生成，须先验收 P2。题池快照、SHA256、抽样 ID 和参数记录在
`runs/p2_dm80_90_v2/protocol.json`；原始三个分片数据另有备份。

```bash
# 原默认条件的冒烟；已发现截断，不应直接启动其全量 P2
python3 scripts/p2_spread.py --config configs/p2_dm80_90_v2.yaml --pool deepmath --band 8.0 --band-width 1.0 --exp p2_dm80_90_v2 --n 5 --graphs hand6 --no-single --concurrency 4
```

本轮默认思考、8,192 token 的冒烟完成 30 格，其中 8 格截断，各图截断率
20%–40%，行间差 20 个百分点，未通过 3 个百分点的门槛。
单题诊断中，默认思考提高至 16,384 token 后仍截断，且遇到两次 502；
`reasoning_effort: low`、16,384 token 同题使用 11,737 token 答对且未截断。
两个诊断配置分别为 `configs/p2_dm80_90_v2_16k.yaml` 和
`configs/p2_dm80_90_v2_low16k.yaml`，实验名与缓存独立。
后者冒烟停止时完成 23/30 格，其中 1 格截断，另有 1 格连续三次 502 后失败，
三次耗时均约 120.01 秒；客户端 `timeout_s` 已为 900。
这提示网关或上游可能存在约 120 秒限制，需查服务端日志确认；降低思考强度未解决问题。
不完整冒烟不作排名或功效结论。结果保存在各实验的 `protocol.json`、原默认条件的
`smoke_report.json` 与 low/16k 的 `smoke_diagnostic.json`；全量 P2 和正式 E0 均未启动。
下一步先排查服务端 502/超时限制，再在独立实验中验证足够的 token 上限，
通过完整冒烟后才启动 150 题全量 P2。同一条件可以缓存续跑；参数变更必须换实验名。
客户端现支持 `model.stream: true`：请求流式输出及最终用量，逐段拼接答案；
缺少结束标记、终止原因或用量时不得写入成功缓存。断流按重试策略处理，已收到的用量
保留在失败调用账本中（未收到的用量仍未知）。该选项也进入实验绑定，重放保持一致。
诊断配置 `configs/p2_dm80_90_v2_stream32k.yaml` 保留默认思考，使用流式传输与 32k 上限，
此前失败的题目已在该条件下一次调用完成：313.58 秒、28,843 token，答案正确且无截断。
这验证了单个长请求可以超过此前约 120 秒的失败点，尚不代表所有图或题目都能通过；
下一步运行同一快照、同一种子的 5 题 × hand6 冒烟，验收后再决定全量 P2。

```bash
python3 scripts/p2_spread.py --config configs/p2_dm80_90_v2_stream32k.yaml --pool deepmath --band 8.0 --band-width 1.0 --exp p2_dm80_90_v2_stream32k --n 5 --graphs hand6 --no-single --concurrency 4
```
提高上限或改变思考强度后的结果不能与原条件合并。旧 P1 数字也不能替代新条件 P2。

`--band-width` 同样适用于下载后的频次检查、`python3 -m data.loaders.deepmath` 和
`data/make_splits.py`；划分文件记录上下界，后续所有驱动按该范围加载。

七步都有驱动和 mock 单测；此前真端点冒烟为 250 格、0 失败、0% 截断
（本轮 DeepMath 新题池的失败情况见上，不代表该历史结果）。历史模型条件：
DeepSeek-V4.1-Flash、温度 0.7、每节点 8192 token、思考用 gateway 默认。

**悬着的一个决定：主格用哪个难度档。** MATH Level 4–5 对 2026 的思考模型已饱和
（单智能体 0.91–0.93），所以换成带连续难度字段的 DeepMath-103K。实测：

| 难度档 | 单智能体 p | 生成长度中位数 | 历史张量工期（旧题池/旧规模/旧条件） |
| --- | --- | --- | --- |
| 7.0 | 0.769 | 1001 | 2.7 天 |
| 8.0 | 0.795 | 2492 | 6.8 天 |
| **8.5** | **0.690** | **1769** | **4.8 天** |
| 9.0 | 0.605 | 6316 | 17 天 |

注意 P1b 脚本默认每档 40 题；若当时用的是默认值，95% CI 约 ±0.14，8.0 与 8.5 的差在噪声内；旧脚本的闭区间还让难度恰为
9.0 的题同时进了两档（已改为半开区间）。下一步跑 P2（`scripts/p2_spread.py`）再决定要不要付全量
工期——决定实验成败的不是题有多难，而是 top 端的拓扑分不分得开、最好的图能不能赢过等 token
的 π_single，以及 val 能不能分辨 ε。
