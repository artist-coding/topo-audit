# 拓扑学习论文 · 文档包

导出日期：2026-10-12。三份文档 + 一个可直接放进仓库的起步骨架。

| 文件 | 内容 | 什么时候用 |
| --- | --- | --- |
| `01-paper-proposal.md` | 论文提案：题目与定位、Abstract（中英）、Introduction、Related Work、Method（四项分解 + 测量协议）、Experiments、风险与算力预算、待定决策 | 写论文、对审稿人问题、决定实验范围 |
| `02-feasibility-and-novelty.md` | 可行性与创新性评估：现有资源（flash 模型 + H20）的精简实验方案、算力预算、时间线、创新性与被攻击点 | 排期、算卡时、决定投哪个 venue |
| `03-experiment-framework-guide.md` | 实验框架搭建指南：架构、仓库结构与构建顺序、核心数据结构与接口、S₀ 定义、执行器与记账、优化器、分析脚本、E0 步骤与硬规则 | **交给本地 Claude Code 搭框架** |
| `starter/CLAUDE.md` | 从指南里提炼的硬规则与构建顺序，放到仓库根目录，Claude Code 会自动读 | 建仓库第一步 |
| `starter/configs/e0.yaml` | E0 的默认配置 | 建仓库第一步 |

## 怎么用

```bash
mkdir topo-audit && cd topo-audit && git init
cp -r /path/to/starter/* .          # CLAUDE.md + configs/e0.yaml
cp /path/to/03-experiment-framework-guide.md docs/
claude    # 然后说：按 docs/03-experiment-framework-guide.md 的第 1 步建 space/，并写单元测试
```

建议一步一个 PR，按指南里"仓库结构与构建顺序"表的 7 步走，每步先过验收再往下。第 1–4 步可以全部用 mock 端点跑通，不占 GPU。

## 两点说明

- **图不在 Markdown 里。** 三份文档在线版各有一张矢量图（Figure 1 四项分解、路线图、框架架构），Markdown 导出时无法带出：Figure 1 在 `01` 里改成了一段文字说明，路线图和架构图在 `02`、`03` 里改成了 ASCII 版。要看原图去在线版。
- **在线版是源。** 三份在线文档仍可编辑和评论，这里是 2026-10-12 的快照：
  - 论文提案：https://claude.ai/code/artifact/0c405e07-850f-4237-b449-101fabd721d5
  - 可行性评估：https://claude.ai/code/artifact/f5afaf9a-7b12-4d9d-878c-9798d450ee98
  - 框架指南：https://claude.ai/code/artifact/06d59d7c-66fc-4bf4-92d0-579185b19506
