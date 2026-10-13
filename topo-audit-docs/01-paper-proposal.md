# What Bounds Topology Learning? — 论文提案

2026-10-12 · 作者：Zx · 在线版：https://claude.ai/code/artifact/0c405e07-850f-4237-b449-101fabd721d5

## 题目与定位

研究问题从"MAS 拓扑能否像参数一样被训练、会不会收敛"（两个答案都已是 yes）改为：**一个已经发生的拓扑学习平台期，离当前搜索空间的最优还有多远？这段距离能否分别归因到基座模型、搜索空间、优化器、任务样本四个因素，并据此决定下一步干预？** 本文定位为分析型（science）论文：不提出新优化器，而是把现有优化器当作被测对象。

理论贡献不在恒等式本身——它是代数上的伸缩和——而在三件事：每一项能否被测量、这些项能否可靠区分一个平台期的真实成因、以及区分结果能否指出下一步该做什么（继续搜、加数据、扩空间，还是停止）。论文的新颖性要落在测量协议、可区分性检验和干预决策上，而不是落在分解公式上。

四个因素具体指什么（正式记号见 3.1，它们与四项的对应见 Figure 1 右侧的点阵）：

| 因素 | 具体是什么 | 本文怎么干预 | 主要驱动哪些项（见 Figure 1 的点阵） |
| --- | --- | --- | --- |
| 基座模型 M | 每个 agent 节点背后调用的那个 LLM（家族、规模、是否带思考模式）。同质空间里所有节点共用同一个 M，异构空间里 M 是一个模型池。拓扑只是在节点之间路由它的能力，所以它定了能力上限 | 换家族、换规模，其余三项固定 | 容量项 C |
| 搜索空间 S | 允许的拓扑集合：有多少个 agent、哪些角色、哪些有向边可以连、每个节点可用的模型/工具/提示，加一个总 token 上限。S 决定"最好的拓扑能有多好" | 逐级扩大：同质无工具 → 异构模型池 → 带检索/代码工具 → 联合提示优化；每级内另设"固定资源、只变拓扑"的对照 | 容量项 C；估计项 E（随 log∣S∣ 增大）；经验优化项 O 的失败模式随 ∣S∣ 变化（交互） |
| 优化器 O | 在 S 里找拓扑的算法：随机搜索、REINFORCE（GPTSwarm 式）、GRPO（Graph-GRPO 式）、MCTS（AFlow 式）、进化（EvoFlow 式）。O 决定"在训练目标上离经验最优有多远"，受奖励噪声影响 | 换算法，评估预算相同 | 经验优化项 O；泛化修正 Δ_gen |
| 任务样本 D_N | 用来训练拓扑的 N 个任务（从任务分布 D 抽样），对应模型训练里的训练集；N 的大小与多样性决定在有限样本上做选择的代价 | N 取 16 / 64 / 256 / 1024；单任务族 vs 混合 | 估计项 E；泛化修正 Δ_gen |

候选题目（推荐第 1 个）：

1. **What Bounds Topology Learning? Attributing the Plateau of Learned Multi-Agent Systems**
2. The Plateau Is Ambiguous: Auditing Where Multi-Agent Topology Learning Stops
3. The Plateau Is Not the Ceiling: Decomposing Convergence in Multi-Agent Topology Learning
4. Where Do Learned Agent Topologies Stop, and Why? Toward a Science of Topology-Learning Asymptotes

一句话 thesis：拓扑学习的平台期本身是歧义的——它可能来自搜索空间的容量上限、按 N 个训练任务做选择造成的估计损失、优化器没达到的经验训练目标，或学到的结构与经验最优结构之间的泛化差异；只有第一项是真正的天花板。本文给出把一个已经发生的平台期分解到这四项上的测量方法，并用计入预算的低成本探测决定下一步该继续搜、加数据、扩空间还是停下。

| 现有工作 | 它们回答什么 | 本文的差别 |
| --- | --- | --- |
| 拓扑/工作流优化器（GPTSwarm、MaAS、AFlow、Graph-GRPO、HieraMAS…） | 怎样更快、更稳地学到更好的拓扑 | 不提出新优化器；把它们当作被测的 O 与 S，测各自卡在哪一项 |
| MAS scaling 与饱和（MacNet、Science of Scaling Agent Systems、Ringelmann） | 手工拓扑族随 agent 数量与模型能力的饱和 | 研究"学出来的拓扑"在训练过程中的渐近线，并把它分解归因 |
| MAS 何时优于单智能体（信息瓶颈视角、Reliability Limits、等算力比较） | 什么机制决定协作是否有益 | 把这些机制陈述转成关于渐近线的可证伪预测，用实验检验 |

本提案采用的关键设定：

- 主设定是**全局拓扑**：在一个任务分布上学一张图（或一个结构分布）；逐 query 生成拓扑的策略作为扩展设定（E8）。原因：收敛定义干净，贴近"参数收敛"类比，也便于穷举 oracle。
- 研究对象是**有限预算平台期**（finite-budget plateau），不是渐近线：预先登记最小有意义提升 ε，用多种子与配对置信区间判断数据能否排除超过 ε 的继续改善；渐近线外推只作探索性分析。
- 单智能体参照是一个**事先声明的策略集合**（CoT、自洽@k、best-of-N + 自校验、自精炼）在相同 token 预算下的最优者；H1 只相对这个集合成立，不声称普遍上界。
- 数据分**四个角色**：训练（给优化器）、oracle 审计（穷举空间内校准）、验证（选超参、预算、预测模型）、最终测试（只用一次）；容量估计必须跨角色报告，避免在同一批数据上"选最高再报最高"的多重比较偏差。
- 所有性能都在相同 token 预算下报告，对应模型训练里的验证 loss。
- 目标 venue：ICLR / NeurIPS / ICML 主会，以 analysis paper 的身份投，不与方法类论文比 SOTA。

## Abstract

方括号里是待实验填入的内容：(i)–(iv) 是研究问题，方括号中的预期结果对应假设 H1–H5，实验后按实际改写。投稿时按 venue 要求压到 200–250 词。

Learning the communication topology of an LLM-based multi-agent system (MAS) from task feedback has become a standard recipe, and these learners are commonly reported to plateau: topologies freeze and held-out accuracy stops improving within the available budget. A tempting reading, borrowed from model training, is that the plateau marks the ceiling of the MAS framework. We argue that this reading is unwarranted and give a way to audit it. We formalize topology learning as a four-factor problem—base model, search space, optimizer, and task sample—and show that the gain of a learned topology over a compute-matched single agent (drawn from a fixed, declared strategy set) decomposes exactly into a capacity term (what the search space can add given the model), an estimation term (the cost of selecting on finitely many training tasks), an empirical optimization term (how far the learner falls short of the empirical optimum on its own training objective), and a signed generalization correction. A plateau is the framework's ceiling only when the last three terms vanish. We make each term measurable with exhaustive audits on enumerable spaces, cross-fitted estimates that never select on the test set, compute-matched single-agent references, and plateau tests with pre-registered minimal effects and confidence intervals. Across [M] models, [S] search spaces, [O] optimizers, and [B] task families ([k] seeds), we find that (i) [x]% of finite-budget plateaus are optimizer- or data-induced rather than capacity-bound; (ii) with homogeneous agents and no external tools, audited capacity exceeds the declared single-agent reference by at most [y] points; (iii) capacity rises only when the search space admits non-redundant information—heterogeneous models or tools—and we separate the topology's own contribution from the resource's; and (iv) [exploratory] stronger models shift learned topologies toward sparser, near-single-agent structures. Finally, low-cost probes run before training predict which intervention—more optimization budget, more tasks, a larger space, or stopping—pays off, within [z] points on unseen model–task pairs, with probe cost charged.

**中文译文**

从任务反馈中学习 LLM 多智能体系统（MAS）的通信拓扑已成为标准做法，而这些学习器常被报告为进入平台期：拓扑冻结，held-out 准确率在可用预算内不再改善。一个从模型训练借来的、很有诱惑力的解读是：平台期标志着 MAS 框架的天花板。我们认为这个解读站不住脚，并给出审计它的方法。我们把拓扑学习形式化为一个四因素问题——基座模型、搜索空间、优化器、任务样本——并证明学到的拓扑相对等算力单智能体（取自一个事先声明的固定策略集合）的收益可以精确分解为四项：容量项（给定模型时搜索空间能增加什么）、估计项（在有限训练任务上做选择的代价）、经验优化项（学习器在自身训练目标上距经验最优的差距）和一个有符号的泛化修正项。只有当后三项为零时，平台期才是框架的天花板。我们用可穷举空间上的穷举审计、从不在测试集上做选择的交叉拟合估计、等算力单智能体参照、以及带预登记最小效应与置信区间的平台期检验，使每一项都可测量。在 [M] 个模型、[S] 个搜索空间、[O] 个优化器和 [B] 个任务族（[k] 个种子）上，我们发现：(i) [x]% 的有限预算平台期来自优化器或数据，而非容量上限；(ii) 当 agent 同质且无外部工具时，审计得到的容量最多只比声明的单智能体参照高 [y] 个点；(iii) 只有当搜索空间容纳非冗余信息（异构模型或工具）时容量才会抬高，且拓扑自身的贡献与资源的贡献被分开度量；(iv)（探索性）模型越强，学到的拓扑越趋向稀疏的、接近单智能体的结构。最后，训练前的低成本探测能预测哪种干预——增加优化预算、增加任务、扩大空间或停止——值得，在未见过的模型–任务对上误差在 [z] 个点以内，探测成本已计入。

## 1. Introduction / Background

引言要在前两段完成一次"反转"：读者默认平台期意味着 MAS 框架到了上限，本文指出这个推论在六处站不住，然后把每一处断裂变成一个可测量的问题。开场用一个具体实例：两个拓扑优化器在同一任务上都进入平台期，但审计显示其中一个离当前空间可达到的最优仍差很远——平台期本身是含糊的（the plateau is ambiguous）。下面是段落级的写作计划。

**§1.1 现状：拓扑学习已成标准配方，而且常被报告为进入平台期。** 从 GPTSwarm 用 REINFORCE 学边概率，到 MaAS 的 agentic supernet，再到 2026 年的 Graph-GRPO、Agent Q-Mix、HieraMAS 和训练时编排的 MAS-Orchestra，拓扑学习的优化器已经有五六种。这些学习器常呈现同一种曲线：先快速提升，然后拓扑冻结、held-out 准确率在预算内不再改善。实践者和部分论文把这个平台期读作"框架的天花板"（引 1–2 处原文作为靶子）。平台期是否真实、是否可重复，本身也是本文要先用统计判据确认的对象，而不是前提。

**§1.2 类比与断裂。** 模型训练里 loss 的平台期有经典分解：近似误差、估计误差、优化误差。把这个分解搬到拓扑学习，会发现六处对不上，每一处都对应平台期的一种替代解释：

- 多出一个冻结的基座模型：拓扑只能路由模型已有的能力，不能凭空创造能力，所以平台期的高度首先是模型的函数。
- 容量不固定：多一个 agent、多一条边就是多一截推理算力，多一个工具就是多一份外生信息；不在等算力、等资源下比较，平台期可能只是 test-time scaling 或工具接入换了个名字。
- 评估带噪且有限：拓扑评估是一个带噪 bandit，候选之间差距越小越难分辨，优化器停下来可能只是分辨不出差异；而且"选最高分的图"本身对噪声做了优化，oracle 若在同一批数据上既选又报，就会高估容量。
- 优化误差与泛化差异混在一起：优化器没达到经验最优，和它找到的结构泛化得更好或更差，是两件事；不拆开，第三项可以为负却被读成"优化器没找到的部分"。
- 收敛对象可能是分布而非一张图，且训练分布外没有泛化保证，平台期可能只是对训练任务的过拟合。
- "收敛"在有限预算内不可证明：曲线平了不等于 t → ∞ 时也平；能检验的只是"数据能否排除超过 ε 的继续改善"。

**§1.3 研究问题。**

- 主问题：为什么 MAS 拓扑学习会出现性能平台期，它距离当前搜索空间可达到的最优性能还有多远？
- RQ1（归因）：一个观察到的有限预算平台期，有多少来自有限训练任务上的选择（估计项）、多少来自优化器在自身训练目标上的差距（经验优化项）、多少来自两者之间的泛化差异？剩下的才是当前搜索空间相对单智能体参照的容量。
- RQ2（边界）：模型能力、任务结构与搜索空间（同质 / 异构 / 带工具）怎样改变可达到的最优性能？工具或异构带来的收益里，有多少属于拓扑本身而不是资源本身？
- RQ3（干预）：能否用训练前的低成本探测，判断继续搜索、增加任务、扩大搜索空间或停止训练哪一种更值得，并把探测成本计入？

**§1.4 贡献。**

1. 一个把拓扑学习平台期精确分解为容量、估计、经验优化、泛化修正四项的恒等式——前三项在穷举审计下非负，第四项有符号——以及让每一项可测量的协议：四种数据角色、交叉拟合的容量估计、等算力的声明式单智能体参照、带预登记 ε 与置信区间的平台期检验。
2. 据我们所知第一个对"学出来的拓扑"的平台期做受控归因的研究：固定三项换第四项并检验因素间交互，覆盖 [M] 个模型、[S] 个搜索空间、[O] 个优化器、[B] 个任务族（含至少一个真实协作任务族）。
3. 两个可证伪的边界命题（相对声明策略集的模型界 H1、分离资源与拓扑贡献的信息界 H2）及其实证检验。
4. 一个由训练前探测驱动的干预决策器（继续搜 / 加数据 / 扩空间 / 停止），在未见模型–任务对上验证，探测成本计入；开源全部学习曲线、拓扑轨迹与审计协议。

**§1.5 Figure 1。** 概念图：四因素如何映射到三项 gap，以及平台期在什么条件下才等于框架的天花板（见 Method 一节的图）。

## 2. Related Work

相关工作分五条线；本文与每一条的关系都是"把它们的对象或结论当作被测量、被检验的东西"，而不是再加一个方法。先给定位表，再给每条线的写作要点。

| 线 | 代表工作 | 它们改变什么 | 本文如何用 |
| --- | --- | --- | --- |
| 2.1 拓扑与工作流学习 | GPTSwarm、AgentPrune、G-Designer、MaAS、ADAS、AFlow、MASS、Graph-GRPO、Agent Q-Mix、HieraMAS、MasFACT | 优化器与搜索空间的设计 | 作为被测的 O 与 S；量各自的优化 gap，不比最终精度 |
| 2.2 MAS 的 scaling 与饱和 | MacNet、More Agents Is All You Need、Science of Scaling Agent Systems、Ringelmann | 手工拓扑族的 agent 数量与模型能力 | 他们的饱和量（单智能体准确率、agent 间相关性）成为本文预测律的自变量 |
| 2.3 MAS 何时优于单智能体 | 信息瓶颈视角、Reliability Limits、等 token 预算比较、BenchAgent、MAST | 协作有益的机制与公平评测协议 | 机制陈述转成可证伪预测 H1/H2；沿用等算力与统一协议 |
| 2.4 学习理论、NAS、推理时算力 | Bottou & Bousquet 分解、DARTS 退化、随机搜索 NAS、test-time scaling | 平台期的经典来源 | 恒等式与等算力参照点的理论依据 |
| 2.5 协作 benchmark 与训练时编排 | MAS-Orchestra / MASBENCH、MASBench（BUPT）、MultiAgentBench、CooperBench | 协作是否真的发生、何时有益、任务结构怎样刻画 | 提供真实协作任务族与任务结构协变量；MAS-Orchestra 是最近的强近邻，差异在"归因已发生的平台期" |

**2.1 拓扑与工作流学习。** 第一代把 agent 当作可优化的计算图：GPTSwarm 用 REINFORCE 学边概率（ICML 2024），DyLAN 用 agent 重要性评分做动态选择，AgentPrune 剪掉冗余通信边并发现大量通信是冗余的，G-Designer 用生成式 GNN 设计任务自适应拓扑，MaAS 把架构建成 agentic supernet、逐 query 采样。代码级工作流一支有 ADAS / Meta Agent Search（元 agent 直接写代码）、AFlow（MCTS 搜索代码表示的工作流）、EvoFlow（进化搜索），以及 MASS——它联合优化提示与拓扑，发现提示层面的优化往往比拓扑搜索贡献更大。2026 年的工作集中在优化器层面：[Graph-GRPO](https://aclanthology.org/2026.findings-acl.1010/) 指出单样本策略梯度配二值奖励方差大、信用分配差，改为对每个 query 采样一组拓扑算组内相对优势；[Agent Q-Mix](https://arxiv.org/html/2604.00344) 把拓扑选择建成合作式 MARL；[HieraMAS](https://www.emergentmind.com/papers/2602.20229) 放弃逐边优化、把拓扑选择当整图分类，并在角色超节点内混合异构模型；[MasFACT](https://www.alphaxiv.org/abs/2605.17361) 进入持续学习设定，提出"拓扑遗忘"并指出此前工作假设任务分布静态。本文不与它们比精度：它们是本文被测的 O 与 S 的实例。

**2.2 MAS 的 scaling 与饱和。** [MacNet](https://arxiv.org/abs/2406.07155) 用 DAG 组织上千个 agent，把节点数从 2^0 扩到 2^6，观察到性能随规模呈 logistic 增长、约百级 agent 饱和，不规则拓扑优于规则拓扑；同篇还报告等 LLM 调用次数下的多数投票几乎不涨（约 +0.9%，八个 agent 即平台）。More Agents Is All You Need 报告采样–投票随 agent 数提升后饱和。[Towards a Science of Scaling Agent Systems](https://arxiv.org/abs/2512.08296v2) 做了受控评测（v1 摘要：180 个配置、4 个 agentic benchmark；v2 摘要：260 个配置、6 个 benchmark × 5 种架构 × 3 个模型家族——引用时按所引版本写）：单智能体基线越过阈值后协作收益递减乃至为负（v1 报告阈值约 45%），工具密集任务受 MAS 开销拖累，无中心化校验的架构更易传播错误，相对单智能体的变化从 +80.8% 到 −70.0%，并能对 87% 的留出配置选出最优架构。[Ringelmann 效应](https://arxiv.org/abs/2606.02646) 给出两参数定律 R(N) = 1/(1 + c(N−1)N^−β)，在 44 个 cell、三个开源家族加 Gemini 校验上所有条件 R² > 0.99；噪声安慰剂与自纠错不可区分，说明同质团队里"辩论"的收益来自重新评估而非同伴内容；只有架构多样性（异构团队）能降低 c、逃离硬上限，通信模式干预不能。本文位置：他们扫描固定拓扑族的规模，本文扫描学习过程的渐近线；他们找到的饱和量——单智能体准确率、agent 间初始相关性——正是本文预测律的候选自变量。

**2.3 MAS 何时优于单智能体。** [信息瓶颈视角](https://www.alphaxiv.org/abs/2607.16133.md) 把 MAS 设计看成信息瓶颈问题，基座越强越不受益于压缩与分工；[Reliability Limits](https://arxiv.org/pdf/2603.26993) 用决策论模型说明新增阶段只在引入外生信号、在通信约束下保住决策相关信息、或提供非冗余审查时才有益；[等思考 token 预算的比较](https://arxiv.org/html/2604.02460v1) 发现算力归一化后单智能体可追平甚至超过 MAS，MAS 只在单智能体上下文利用退化或投入更多算力时才有竞争力；[BenchAgent](https://arxiv.org/pdf/2606.05670) 把单智能体、固定 MAS、演化 MAS 放在同一套加载器、工具权限、计费与日志协议下比较；MAST（Why Do Multi-Agent LLM Systems Fail?）给出失败分类，多数失败来自规格与协调。本文位置：把这些机制陈述转成关于渐近线的可证伪预测（H1 模型界、H2 信息界），并沿用等算力与统一协议的评测标准。

**2.4 学习理论、NAS 与推理时算力。** Bottou & Bousquet（2008）的近似、估计、优化三项分解是本文恒等式的来源。NAS 里有现成的前车之鉴：DARTS 类超网会退化到平凡结构（Zela et al., 2020），随机搜索常与复杂搜索相当（Li & Talwalkar, 2019；Yu et al., 2020）——"优化器导致的平台期"在 NAS 中已被反复观察，而 MaAS 的 supernet 直接继承了这一风险。推理时算力一线（Snell et al., 2024 的 compute-optimal test-time scaling；Brown et al., 2024 的 Large Language Monkeys 覆盖率幂律；Wu et al., 2024 的 inference scaling laws）是本文等算力单智能体参照的依据。

**2.5 协作 benchmark 与训练时编排（新增）。** [MAS-Orchestra](https://arxiv.org/abs/2601.14652)（ICML 2026）把 MAS 编排建成函数调用式强化学习问题、一次生成整个 MAS，并提出控制型 benchmark MASBENCH，按 Depth、Horizon、Breadth、Parallel、Robustness 五个轴刻画任务，用"degree of MAS"分析协作何时相对单智能体有价值。这是本文最近的强近邻：它回答"训练出的编排何时有益"，本文回答"一个已经发生的平台期由什么限制"，并把它的五个轴当作任务结构协变量。真实协作任务方面：[MultiAgentBench](https://aclanthology.org/2025.acl-long.421/)（ACL 2025，MARBLE 框架）在协作与竞争场景下比较 star、chain、tree、graph 协调协议，用里程碑式指标衡量协作质量；[MASBench](https://arxiv.org/abs/2610.04672)（BUPT，2026-10，与 MAS-Orchestra 的 MASBENCH 重名）在部分可观测约束下按 Reasoning、Scheduling、Game 三类任务评测 Protocol、Memory、Routing 三种协作机制，并给出通信成本与性价比指标；[CooperBench](https://github.com/cooperbench/CooperBench)（652 个任务、12 个仓库）让两个编码 agent 在有潜在冲突的功能上协作，发现同等总工作量下协作 agent 明显差于单 agent。本文位置：主实验至少包含一个真实协作任务族，避免论文退化为"不同 LLM 在不同计算图上的准确率"；同时不把"模型越强，MAS 越不值得"作为主贡献，这类结论已有近邻。

## 3. Method

方法部分不提出新优化器，而是给出一个恒等式和一套测量协议：把学到的拓扑相对等算力单智能体的收益精确拆成四项，再用四因素干预分别移动它们。

**3.1 Setup and notation.** 一个 MAS 是带属性的图 G = (V, E, ρ, π, μ)：agent 集合 V、有向通信边 E、角色 ρ、提示 π、模型分配 μ。搜索空间 S 是在算力上限 c（总 token）下允许的 G 的集合。基座模型 M（异构空间下是一个模型池），任务分布 D，训练样本 D_N（N 个任务），以及审计、验证、最终测试三个互不重叠的独立样本。拓扑学习器 (O, t) 用优化器 O、评估预算 t 输出 G_{O,t}。两种性能要分开：总体性能 P(G) 是 D 上、算力预算 c 下的期望得分；经验性能 P̂_N(G) 是 D_N 上的平均得分，也是优化器实际最大化的目标。学习曲线 L(t) = P(G_{O,t})。**研究对象是有限预算平台期**：预先登记最小有意义提升 ε；在预算 t 内，若 ≥ 5 个种子的配对置信区间能排除后续 Δt 预算带来超过 ε 的改善，则称学习器在 t 处出现平台期。"最近几轮变化不显著"不等于"没有改善"，所以判据是排除而不是未检出。渐近线外推（L(t) = P_O − a·t^{−b} 或 logistic）只作探索性分析，不作主结果：有限预算内的停滞不等于 t → ∞ 的收敛，换拟合形式、提高候选评估精度或改变探索率都可能改变外推值。主设定下 G 是分布 D 上的一张图或一个结构分布；扩展设定下 G = g_θ(x) 逐 query 生成，分解对策略同样成立。

**3.2 平台期分解（核心恒等式）。** 五个量：

| 量 | 定义 | 怎么得到 |
| --- | --- | --- |
| P_single | 事先声明的单智能体策略集合（CoT、自洽@k、best-of-N + 自校验、自精炼）在同一模型、同样 token 预算下的最优者的总体性能 | 验证集选策略，最终测试集报告 |
| P(G*) | S 中总体最优拓扑 G* 的性能，即当前空间的容量上限 | 可穷举空间：审计集交叉拟合选出、独立样本上估计并报区间；大空间：高预算参照只给下界 |
| P(Ĝ_N) | 训练样本上经验最优的拓扑 Ĝ_N 的总体性能 | 小空间穷举经验最大化，独立样本评估 |
| P̂_N(·) | 任一拓扑在 N 个训练任务上的经验性能，即优化器的真实目标 | 直接计算 |
| P(G_{O,t}) | 优化器 O 用预算 t 输出的拓扑的总体性能 | 独立样本评估 |

**直观版（先看数字，再看公式）。** 设审计空间 S₀ 只有 64 张图，基座模型固定，训练题 256 道（D_N），另有一批从不参与选择的测试题。两种分数要分开：真实分数 P(G) 是一张图在整个任务分布上的期望准确率，用独立测试题估；训练分数 P̂_N(G) 是同一张图在 256 道训练题上的准确率，也是优化器实际最大化的目标。四个角色：P_single——同模型、同预算、不搭 MAS 时的最好成绩，设 60 分；G*——64 张图里真实分数最高的那张，设 68 分，它是这个空间的天花板；Ĝ_N——64 张图里训练分数最高的那张，训练分 70、真实分 66（训练题太少，"训练集上最好"选中了一张有点过拟合的图）；G_O——优化器跑完预算交出的图，训练分 67、真实分 65。

| 项 | 算法 | 对收益的贡献 | 含义 |
| --- | --- | --- | --- |
| 容量 C | P(G*) − P_single = 68 − 60 | +8 | 这个空间最多能比单智能体多给 8 分 |
| 估计 E | P(G*) − P(Ĝ_N) = 68 − 66 | −2 | 训练题太少、选错了图，丢 2 分 |
| 经验优化 O | P̂_N(Ĝ_N) − P̂_N(G_O) = 70 − 67 | −3 | 优化器在自己的训练目标上没搜到头，丢 3 分 |
| 泛化修正 Δ_gen | (66 − 70) − (65 − 67) = −2 | +2 | Ĝ_N 过拟合 4 分、G_O 只过拟合 2 分，反而赚回 2 分 |

8 − 2 − 3 + 2 = 5，正好等于 P(G_O) − P_single = 65 − 60。优化器停在 65 分，看起来像天花板，但天花板是 68：3 分是优化器没搜到，2 分是数据不够，另有 2 分是它因为找到了泛化更好的图赚回来的。旧版三项分解会把 O 与 Δ_gen 合成一项"−1 分"，读成"优化器丢了 1 分"——其实是"丢了 3 分又赚回 2 分"，两种诊断对应完全不同的下一步（3.6）。下面的恒等式就是这张账单的一般形式。

```latex
\underbrace{P(G_{O,t}) - P_{\mathrm{single}}}_{\text{achieved gain}}
= \underbrace{P(G^{*}) - P_{\mathrm{single}}}_{C\ (\text{capacity})}
- \underbrace{\big(P(G^{*}) - P(\hat G_N)\big)}_{E\ (\text{estimation})}
- \underbrace{\big(\hat P_N(\hat G_N) - \hat P_N(G_{O,t})\big)}_{O\ (\text{empirical optimization})}
- \underbrace{\Big(\big[P(\hat G_N) - \hat P_N(\hat G_N)\big] - \big[P(G_{O,t}) - \hat P_N(G_{O,t})\big]\Big)}_{\Delta_{\mathrm{gen}}\ (\text{signed generalization correction})}
```

恒等式是伸缩和，无需任何假设。E 与 O 在同一搜索空间、精确经验评价下都非负：Ĝ_N 按定义不会优于总体最优 G*，G_{O,t} 按定义在训练目标上不超过经验最优 Ĝ_N。Δ_gen 有符号：优化器若借早停、正则或搜索偏好找到了泛化更好的结构，Δ_gen 为负——这不是"负的优化误差"，而是泛化差异。旧版三项分解把 O 与 Δ_gen 混在第三项里（P̂_N − P_O 可以为负却被读成"优化器没找到的部分"），是评审指出的主要漏洞，现已拆开。读法：平台期等于"框架的天花板"当且仅当 E、O、Δ_gen 都约为零，此时 P(G_{O,t}) ≈ P(G*)，往上走只剩扩大 S 或换更强的 M。C 可以为负——拓扑在等算力下打不过声明的单智能体参照。大空间上拿不到 P(G*)：只报告不依赖它的 O 与 Δ_gen，C 与 E 给区间，精确四项分解只在可穷举空间上声称。

> **Figure 1（在线版为矢量图）** 左边是学到的拓扑相对等算力单智能体参照的收益，中间四项是它的精确分解——容量、估计、经验优化、泛化修正——右边的点阵标出每个因素主要驱动哪些项（实心为主效应，空心为交互，不是一一对应）；只有容量项才是"框架的天花板"。

**3.3 测量协议（每一项怎么估）。**

- 数据角色：训练集（给优化器学拓扑）、oracle 审计集（穷举空间内校准各结构）、验证集（选超参、预算、预测模型）、最终测试集（只用于最终报告）。四者互不重叠。
- 容量 C 的估计要避开选择偏差：在同一批数据上取 max 会对噪声做选择，max_G P̂_audit(G) 系统性高于 max_G P(G)。做法是交叉拟合——审计集分两半，在一半上选 G*，在另一半上评估，轮换取平均；报告的 P(G*) 来自最终测试集并附 bootstrap 区间。即使 S_0 可以完全穷举，拿到的也只是有限样本上的最优图，总体最优仍是带区间的统计估计。大空间 S_1 的高预算参照（所有优化器最优解 ∪ 10 倍预算随机搜索）只给 C 的下界，不得当作上限。
- 估计 E：在 S_0 上对 N ∈ {16, 64, 256, 1024} 穷举经验最大化得 Ĝ_N，在独立样本上评估得 P(G*) − P(Ĝ_N) 随 N 的曲线；配一个有限空间的一致收敛界作参照（|S_0| 个候选、N 个样本，形如 √(log|S_0| / N)），看经验曲线是否贴着界走。
- 经验优化 O 与泛化修正 Δ_gen：按定义直接算，每个优化器、每个预算 t 各一条曲线；O 用训练集上的经验性能，Δ_gen 用独立样本。
- 可分辨性诊断（在线用，不用于训练前预测）：同一批任务上两个候选拓扑的配对奖励差及其标准误——任务难度在拓扑间高度相关，配对比无配对敏感得多。它随预算的轨迹用于解释优化器何时停住。不设普适的 SNR 停止阈值：噪声可以靠重复评价压低，阈值随优化器而变。
- 等算力：所有 P 在相同总 token（含编排开销）下报告，给出前沿而非单点；单智能体参照拿同样的预算。
- 噪声与平台期检验：每个学习器 ≥ 5 个种子，温度固定；平台期按 3.1 的预登记 ε 与配对置信区间判定，只有能排除 > ε 的改善时才宣布。

**3.4 四因素干预设计（固定三项换第四项）。** 因素与分解项不是一一对应：|S| 同时进入 C 和 E，O 的失败模式随 |S| 变化，D_N 与 O 共同决定 Δ_gen。主效应之外报告两两交互（O × |S|、M × S、N × |S|）。

| 因素 | 水平 | 固定项 | 主要移动哪些项 | 验证的假设 |
| --- | --- | --- | --- | --- |
| 基座模型 M | 2 个开源家族 × 小/中/大 + 1 个前沿 API 模型 | S_0、O = GRPO、N = 256 | C；拓扑形态（探索性） | H1、H5 |
| 搜索空间 S | S_homog ⊂ S_hetero ⊂ S_tool ⊂ S_prompt；每个资源条件内另设"固定一个平凡拓扑、只加资源"的对照，把资源收益与拓扑收益分开 | M_mid、O = GRPO、N = 256 | C；E（随 log∣S∣ 增大） | H2 |
| 优化器 O | 随机搜索、REINFORCE（GPTSwarm 式）、GRPO（Graph-GRPO 式）、MCTS（AFlow 式）、进化（EvoFlow 式） | M_mid、S_0、N = 256 | O；Δ_gen | H3 |
| 任务样本 D_N | N ∈ {16, 64, 256, 1024}；单任务族 vs 混合；跨任务族迁移 | M_mid、S_0、O = GRPO | E；Δ_gen | H4 |

**3.5 可证伪的假设。**

- H1（相对声明参照的模型界，条件性经验假设）：当 S = S_homog、且 P_single 取自声明的策略集合（含独立采样与自校验）时，审计容量 C ≤ ε。它不是普遍定理——一个编排能力充分的单控制器原则上可以用同一模型模拟任何 MAS 工作流，所以"最佳单智能体"取决于允许它做什么——因此只相对事先写明的策略集合与相同预算成立。
- H2（信息界，分离资源与拓扑）：从 S_homog 到 S_hetero、S_tool 的容量增量里，把"资源本身的收益"（固定一个平凡拓扑、只加资源）与"拓扑在新资源下的额外收益"分开；预测后者由 agent 间输出相关性的下降（Kish 设计效应）解释，新增 agent 冗余时为零。第三组因为有检索而更好，不等于学到了更好的通信结构。
- H3（优化器可分辨性，在线诊断）：经验优化项 O 的大小由"配对奖励差的标准误相对候选间真实差距"解释；GRPO 与 MCTS 缩小但不消除 O；增加重复评价能降低 O，说明它不是固定阈值现象。
- H4（数据界）：估计项 E 随 N 按接近 √(log|S| / N) 的速率下降；任务多样性比数量更能移动它；学到的拓扑跨任务族迁移差（拓扑过拟合）。
- H5（探索性，不预设结论）：模型越强，Ĝ_N 是否越稀疏、越趋向单智能体加校验者；前人报告的饱和阈值是否对应 C 过零的位置。作为分析结果报告，不进入主张。

**3.6 从预测律到干预决策。** 分两个层次。在线诊断：训练过程中可得的量（配对奖励差、经验优化项的轨迹、训练–验证差）用来判断当前平台期属于 O、E 还是 Δ_gen。训练前预测：只用训练开始前就能拿到的低成本探测——模型在验证集上的单智能体准确率 p、少量候选拓扑的配对奖励差（相当于 Ringelmann 式的小 N 试点）、agent 间初始输出相关性 ρ₀、是否有外生信息、log N——回归到四项与总收益上，候选形式 C = β₀ + β₁·p(1 − p) + β₂·(1 − ρ₀) + β₃·𝟙[外生信息]，允许交互项。输出不是一条曲线，而是一个决策：继续增加优化预算、增加独立任务、扩大合法拓扑空间，或停止训练；在留出的模型与任务族上检验决策准确率与收益的 MAE。探测本身消耗的 token 计入总预算，与"直接多搜一会儿"做等算力比较。

## 4. Experiments

分阶段进行，不一开始就铺满矩阵：先做 E0 pilot 证明核心测量能工作，再做 E1–E2 确认现象并分解，然后 E3–E5 做单因素干预与交互，最后 E6 做干预决策；E7 排除提示词混淆，E8 把分解推广到逐 query 策略。E0 做不出来，就不扩模型。

**4.1 Setup。**

- 任务族（4 个，覆盖"无外生信息 / 需要外生信息 / 真实协作"三类；具体版本实施时冻结；每个任务族都切出训练 / 审计 / 验证 / 最终测试四个不重叠样本，并记录所用模型的发布日期以防污染）：

| 任务族 | 类型 | 为什么选 | 数据与划分 |
| --- | --- | --- | --- |
| 数学推理 | 封闭答案、无外生信息 | H1 的主战场：拓扑只能路由能力 | MATH-500 只有 500 题，不能同时供训练与测试：训练 / 审计 / 验证取自 MATH 训练集（7,500 题）抽样，MATH-500 只做最终测试；AIME 近年题作外推检查 |
| 代码 | 可执行验证 | 校验者角色有真实价值，测退化方向 | LiveCodeBench 时间窗口切片：早期窗口做训练 / 审计 / 验证，晚于所有模型发布日期的窗口做最终测试 |
| 检索问答 | 需要外生信息 | H2：S_tool 应抬高容量 | FRAMES 全集只有 824 题：训练 400 / 审计 100 / 验证 100 / 最终测试 224；若训练不足，训练与审计改用 MuSiQue 多跳子集，FRAMES 只做最终测试 |
| 真实协作 | 部分可观测、角色依赖、需要通信 | 证明拓扑改变的是协作过程而不只是采样形式 | MultiAgentBench 协作场景（MARBLE）或 MASBench（部分可观测）的 Reasoning / Scheduling 子集，二选一作主任务族；CooperBench 的 coop 设定作补充；MAS-Orchestra 的 MASBENCH 五轴作任务结构协变量 |

- 模型：2 个开源家族各三档（候选 Qwen3 系列、gpt-oss 系列）+ 1 个前沿 API 模型；版本在实施时冻结并写入附录。
- 搜索空间：S_0（可穷举，≤ 4 agent、固定角色集、DAG）、S_1（更大）、S_hetero、S_tool、S_prompt。
- 优化器：随机搜索、REINFORCE、GRPO、MCTS、进化，评估预算相同，统一接口与日志（借 BenchAgent 的协议）。
- 算力：所有比较在相同总 token 下，报告前沿；单智能体参照拿同样预算。
- 种子：每个学习器 5 个；渐近线 bootstrap 置信区间；温度固定。

**4.2 实验表。**

| 实验 | 问题 / 假设 | 变化 | 固定 | 测量 | 预期图 |
| --- | --- | --- | --- | --- | --- |
| E0 Pilot：平台期审计 | 核心测量能否工作 | 随机搜索 + 1 个训练型优化器 | 1 个开源模型、固定 4 节点约 64 个合法 DAG、1 个可自动评分的任务族、3–5 种子 | 学到的结构、经验最优结构、oracle 参照结构各自的性能与 CI；四项分解；交叉拟合与单次 max 的差 | Fig 1b：一个平台期的归因实例 |
| E1 平台期存在吗 | 现象确认 | 5 个优化器的学习曲线 | M_mid、S_0、N = 256 | 预登记 ε 的平台期检验、CI 宽度；渐近线外推只作探索 | Fig 2：带 CI 的学习曲线 |
| E2 分解 | RQ1 | 优化器 | M_mid、S_0 | P(G*)（交叉拟合）、P(Ĝ_N)、O、Δ_gen 及各自 CI；非容量占比 | Fig 3：按 (M, O) 的堆叠条 |
| E3 模型干预 | H1、H5 | 7 个模型 | S_0、GRPO、N = 256 | 等算力下 C 相对声明参照的符号与大小；收敛拓扑形态（探索性） | Fig 4：C vs p 的过零点；Fig 5：拓扑解剖 |
| E4 搜索空间干预 | H2 | S_homog ⊂ S_hetero ⊂ S_tool；每个资源条件内固定平凡拓扑的对照 | M_mid、GRPO、N = 256 | 资源收益 vs 拓扑在新资源下的额外收益；后者 vs 1 − ρ₀ | Fig 6：增量分解与相关性散点 |
| E5 数据干预 | H4 | N、多样性、跨任务族 | M_mid、S_0、GRPO | E 与 Δ_gen 随 N；对照 √(log∣S₀∣ / N) 的一致收敛界；跨任务族迁移矩阵 | Fig 7：E 随 N 的曲线与界；Fig 8：迁移热图 |
| E6 干预决策 | RQ3 | 拟合格子 vs 留出格子 | — | 训练前探测 → 决策准确率、收益 MAE；探测成本计入并与"直接多搜"等算力比较 | Fig 9：预测 vs 观测；决策混淆矩阵 |
| E7 提示词对照 | MASS 的质疑 | S_prompt 重做 E2 | M_mid | 联合提示优化后四项占比是否变化 | 附录表 |
| E8 逐 query 策略 | 分解是否推广 | MaAS / Graph-GRPO 式策略 | M_mid、S_1 | 策略层面的四项 | 附录图 |

**4.3 分析。**

- 收敛拓扑解剖（探索性）：边数、深度、是否有环、agent 数、角色使用率随模型强度的变化（接 H5）。
- 可分辨性分析：配对奖励差的标准误与各优化器停止点的关系（接 H3）；在 S_0 上与 oracle 对照，给出 O 的精确数。
- 交互：因素两两交互（O × |S|、M × S、N × |S|）的效应量，不默认每一项只受一个因素影响。
- 成本：收敛时每任务 token 与单智能体对比；探测成本与"直接多搜"的等算力比较。
- 消融：温度、每个候选的评估样本数、每个优化器的预算。

**4.4 指标。** 等算力的最终测试准确率；平台期检验（预登记 ε、配对 CI）；四项 C、E、O、Δ_gen 及 bootstrap 区间；agent 间输出相关性（成对一致率 / Kish 设计效应）；配对奖励差及其标准误；拓扑稀疏度；跨任务族迁移矩阵；干预决策准确率、收益 MAE 与探测成本。

**4.5 主要结论的三种可能结局（都可发表，写法不同）。**

1. 容量几乎为零、平台期主要来自优化器与数据：标题改为"The Plateau Is Not the Ceiling"，主图是 Fig 3。
2. 容量显著但被模型界住、异构/工具才能抬高：标题保留 "What Bounds Topology Learning?"，主图是 Fig 4 + Fig 6。
3. 容量随模型增强而消失（强模型下退化到单智能体）：主图是 Fig 4 的过零点，与 Science of Scaling 的饱和阈值对账。

## 风险、审稿人预期问题与算力预算

最大的风险不是结果不好看，而是四项 gap 测不准（CI 太宽）或搜索空间被质疑"刻意弱"；两者都要在 pilot 阶段先排除。

| 审稿人会问 | 预案 |
| --- | --- |
| 第三项为什么不是优化误差？ | 旧三项分解确有此问题（P̂_N − P_O 可为负）；现拆成经验优化项 O（训练目标上、非负）与有符号的泛化修正 Δ_gen。 |
| oracle 是否泄漏测试集？ | 四种数据角色互不重叠；容量用审计集交叉拟合选、最终测试集报告；有限样本上的 max 会选中噪声，所以 P(G*) 只以区间报告，并在 pilot 里量化单次 max 的高估幅度。 |
| 平台期是真的收敛吗？ | 主结果只声称有限预算平台期（预登记 ε + 配对 CI，判据是排除而非未检出）；渐近线外推降为探索性分析。 |
| 与 Towards a Science of Scaling Agent Systems 有何不同？ | 他们扫描 5 种手工架构随 agent 数与模型能力的变化；本文审计学习过程的平台期并做四项分解。他们的饱和阈值在本文里是被检验的命题（H5）和预测律的自变量。 |
| 与 MAS-Orchestra 有何不同？ | 它训练编排器并用控制型 benchmark 回答"协作何时有益"；本文不训练新编排器，而是审计任何学习器已经出现的平台期并归因；它的五个任务轴是本文的协变量。 |
| 与 Ringelmann / 信息瓶颈视角有何不同？ | 他们在固定拓扑（全连接辩论、固定分工）下解释饱和机制；本文把机制陈述变成对"学出来的拓扑"平台期的定量预测（H1、H2）并实证检验。 |
| oracle 只在小空间可得，结论能推广吗？ | 小空间给精确校准，大空间的高预算参照只给 C 的下界并如此报告；恒等式本身与空间大小无关，但精确四项分解只在可穷举空间上声称。 |
| 等算力比较公平吗？ | 统一 token 计费、含编排开销，给前沿而非单点；单智能体参照是事先声明的策略集合（含 best-of-N、自洽、延长推理、自精炼）中最强者。 |
| 工具带来的提升是拓扑的功劳吗？ | 每个资源条件内另设固定平凡拓扑的对照，把资源收益与拓扑收益分开（H2）。 |
| 平台期是不是提示词没调好？ | E7 用联合提示优化的 S_prompt 重做分解；若四项占比不变，结论与提示无关。 |
| 搜索空间是否刻意弱？ | S 一路从同质扩到异构与工具；若 S_tool 下容量仍为零，才下"拓扑无用"的结论。 |
| 只是 LLM scaling，不是真正的协作？ | 主实验含一个真实协作任务族（MultiAgentBench / MASBench），报告通信与角色依赖的指标。 |
| benchmark 污染？ | 选时间窗口型或近期集合，报告 train–test 一致性与模型发布日期；MATH-500 只做最终测试。 |
| 只是一篇 negative result？ | 三种结局（4.5）都有正向产出：审计协议、干预决策器；且"平台期不是天花板"本身就是对一条热门赛道的纠偏。 |

**主要风险。**

- 四项 gap 的置信区间太宽：对策是在 S_0 上把每张图的评估次数提高到 CI 可控，再决定大空间的预算；E0 的通过判据就是"四项能区分至少两种成因"。
- oracle 选择偏差：审计集上的单次 max 会高估容量——交叉拟合 + 独立测试集报告，并在 pilot 中量化偏差大小。
- 不同优化器实现差异大、预算不可比：统一评估预算（候选评估次数 × 每次任务数）与接口，先跑随机搜索做基线。
- 成本失控：按下表的杠杆先做 pilot，再决定全量。
- 拓扑学习在强模型上整体退化、所有格子容量为零：这是结局 1/3，仍可发表，但要在摘要里把 claim 改成 negative result + 干预决策器。

**算力预算（数量级估计；假设每次 MAS 运行约 6k token，4 个 agent）。**

| 组件 | 规模假设（算式写明，便于复核） | 约 token 数 | 压缩杠杆 |
| --- | --- | --- | --- |
| E0 Pilot | 64 图 × 300 任务 × 2 次 = 3.84 万次；加 2 优化器 × 5 种子 × 200 次候选评估 × 24 任务 = 4.8 万次；共 8.6 万次 × 6k | 约 0.5B | 先跑这个 |
| S_0 oracle | 150 图 × 200 任务 × 2 次 × 3 开源模型 × 4 任务族 = 72 万次 × 6k | 约 4.3B | 图数降到 100；测试任务 150 |
| E1/E2 优化器扫描 | 5 优化器 × 5 种子 × 300 次候选评估 × 24 任务 × 4 任务族 = 72 万次 × 6k，M_mid | 约 4.3B | 非核心任务族用 3 种子 |
| E3 模型扫描 | 6 模型 × 3 种子 × 300 × 24 × 2 任务族 = 25.9 万次 × 6k | 约 1.6B + API 费用 | API 模型只跑 1 种子 |
| E4 搜索空间扫描 | 3 额外空间 × 5 种子 × 300 × 24 × 2 任务族 = 21.6 万次 × 8k（工具使 token 增加） | 约 1.7B | S_tool 只在检索问答上跑 |
| E5 数据扫描 | 4 个 N × 5 种子 × 300 × 24 × 2 任务族 = 28.8 万次 × 6k | 约 1.7B | N 只取 3 档 |
| E7/E8 | 提示词对照与逐 query 策略 | 约 2B | 放附录，1 任务族 |

合计约 16B token（含 pilot）。按开源模型在 H100 级卡上约 2–4k token/s 的吞吐粗算，约 1,500–3,000 GPU 小时，另加 API 费用约 $3k–8k（取决于前沿模型定价）。作为参照，Ringelmann 一文报告全项目约 2,000 GPU 小时。先用 E0（约 0.5B token、2–3 周）验证四项可测、CI 可控、交叉拟合有效，再决定全量。

> 注：实际执行按《可行性与创新性评估》里的精简方案（无付费 API、合计约 8B token）走，本表是全量版的上界。

## 待定决策与下一步

五个决定影响实验设计，其中第一个决定了 3.1 的收敛定义和 E8 的地位，建议最先拍板。

- [ ] 主设定：全局拓扑（本提案默认，推荐）还是逐 query 策略？后者"收敛"的定义复杂一个量级。
- [ ] 预登记：最小有意义提升 ε、平台期判据、四种数据角色的切分、声明的单智能体策略集合，写成 pre-registration 文档。
- [ ] E0 Pilot（2–3 周）：1 个开源模型；固定 4 节点执行顺序、6 条可选前向边 → 2^6 = 64 个组合，剔除不满足输入输出约束的非法图并决定是否允许孤立节点；随机搜索 + 1 个训练型优化器；1 个可自动评分的任务族；3–5 个种子。产出：对一个观察到的平台期，展示学到的结构、经验最优结构、oracle 参照结构各自的性能与 CI，以及四项归因。
- [ ] E0 通过判据：四项的 CI 能区分至少两种成因（如 O ≫ E）；交叉拟合的容量估计与单次 max 的差能被量化；平台期检验在 5 个种子下能给出结论。
- [ ] 模型清单：2 个开源家族各三档 + 1 个前沿 API，版本冻结并写入附录。
- [ ] 任务族清单：数学（MATH 训练集抽样 / MATH-500 测试）、代码（LiveCodeBench 时间窗）、检索问答（FRAMES 824 题切分或 MuSiQue 训练）、真实协作（MultiAgentBench / MASBench 二选一）。
- [ ] E0 通过后：画出 Fig 1b 与 Fig 2 的初版，决定 venue 与 deadline，再启动 E1–E6。

## Sources

- [Graph-GRPO: Stabilizing Multi-Agent Topology Learning via Group Relative Policy Optimization (ACL Findings 2026)](https://aclanthology.org/2026.findings-acl.1010/)
- [HieraMAS: Optimizing Intra-Node LLM Mixtures and Inter-Node Topology for Multi-Agent Systems](https://www.emergentmind.com/papers/2602.20229)
- [Agent Q-Mix: Selecting the Right Action for LLM Multi-Agent Systems through Reinforcement Learning](https://arxiv.org/html/2604.00344)
- [MasFACT: Continual Multi-Agent Topology Learning via Geometry-Aware Posterior Transfer](https://www.alphaxiv.org/abs/2605.17361)
- [MAS-Orchestra: Understanding and Improving Multi-Agent Reasoning Through Holistic Orchestration and Controlled Benchmarks (ICML 2026)](https://arxiv.org/abs/2601.14652) · [ICML 页面](https://icml.cc/virtual/2026/poster/66445)
- [Scaling Large Language Model-based Multi-Agent Collaboration (MacNet, ICLR 2025)](https://arxiv.org/abs/2406.07155)
- [Towards a Science of Scaling Agent Systems (v2)](https://arxiv.org/abs/2512.08296v2) · [v1](https://arxiv.org/abs/2512.08296v1)
- [The Ringelmann Effect in Multi-Agent LLM Systems: A Scaling Law for Effective Team Size](https://arxiv.org/abs/2606.02646)
- [When Do Multi-Agent Systems Help? An Information Bottleneck Perspective](https://www.alphaxiv.org/abs/2607.16133.md)
- [On the Reliability Limits of LLM-Based Multi-Agent Planning](https://arxiv.org/pdf/2603.26993)
- [Single-Agent LLMs Outperform Multi-Agent Systems on Multi-Hop Reasoning Under Equal Thinking Token Budgets](https://arxiv.org/html/2604.02460v1)
- [Do More Agents Help? Controlled and Protocol-Aligned Evaluation of LLM Agent Workflows (BenchAgent)](https://arxiv.org/pdf/2606.05670)
- [MultiAgentBench: Evaluating the Collaboration and Competition of LLM agents (ACL 2025)](https://aclanthology.org/2025.acl-long.421/)
- [MASBench: Benchmarking LLM-based Multi-Agent Collaboration under Partial Observability (BUPT, 2026-10)](https://arxiv.org/abs/2610.04672)
- [CooperBench: Why Coding Agents Cannot be Your Teammates Yet](https://github.com/cooperbench/CooperBench) · [arXiv](https://arxiv.org/abs/2601.13295)
