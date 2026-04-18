# Inside Hermes Agent：架构与设计哲学

> 对 Hermes Agent（Nous Research）开源代码的深度源码剖析与架构解读。

---

英文版说明请参阅 [README.md](./README.md)。

---

## 目录

- [Inside Hermes Agent：架构与设计哲学](#inside-hermes-agent架构与设计哲学)
  - [目录](#目录)
  - [引言](#引言)
  - [核心架构支柱](#核心架构支柱)
    - [1. 上下文管理](#1-上下文管理)
    - [2. 意图路由](#2-意图路由)
    - [3. 自进化机制](#3-自进化机制)
    - [4. 工具系统](#4-工具系统)
  - [文档目录](#文档目录)
  - [核心发现](#核心发现)

---

## 引言

**Hermes Agent** 是由 [Nous Research](https://github.com/nousresearch/hermes-agent) 开发的自进化（Self-Improving）AI Agent，定位为"通用、自进化、跨平台"的 AI 助手。它不仅仅是对 API 的简单封装——而是一个精密的多 Agent 系统，设计用于在本地文件系统、远程环境和多种消息平台（Telegram、Discord、Slack、WhatsApp、Signal）上自主运行。

通过分析其源代码，本项目提取底层的 Python 结构，揭示 Hermes 如何解决 AI Agent 开发中最棘手的问题：**上下文窗口限制、长期记忆保留、动态工具注册、多层意图路由，以及模型级能力进化。**

---

## 核心架构支柱

### 1. 上下文管理

Hermes Agent 的上下文管理方案采用了**分层解耦**的设计思路，将上下文管理拆分为多个职责明确的独立模块：

- **系统提示词构建层（`prompt_builder.py`）：** 组合 Agent 身份、平台提示、Memory/Skills 指导文本、项目上下文文件，以及两级缓存的 Skills 索引（进程内 LRU + 磁盘快照）。
- **上下文引用层（`context_references.py`）：** 支持 `@file`、`@folder`、`@diff`、`@git`、`@url` 语法，带严格的 Token 预算控制（50% 硬限制，25% 软限制）和双重沙箱（路径边界 + 敏感路径拦截）。
- **上下文压缩层（`context_compressor.py`）：** 四阶段管道：工具输出裁剪 → Head/Tail 边界保护 → LLM 结构化摘要 → 工具对完整性修复。迭代压缩保留前次摘要，避免信号衰减。
- **Prompt Caching 层（`prompt_caching.py`）：** `system_and_3` 策略——固定 4 个 `cache_control` 断点（系统提示词 + 最近 3 条非系统消息），减少约 75% 的输入 Token 成本。
- **消息格式适配层（`anthropic_adapter.py`）：** OpenAI ↔ Anthropic 格式双向转换，严格角色交替强制执行，以及 OAuth 兼容性处理（模拟 Claude Code 身份）。
- **辅助路由层（`auxiliary_client.py`）：** 辅助任务（压缩/LLM 调用）的统一 Provider 解析链——OpenRouter → Nous → Custom → Codex → Anthropic → 直连 API-Key。

📄 **完整报告：** [上下文管理.pdf](./ZH/上下文管理.pdf)

---

### 2. 意图路由

Hermes Agent 的**意图路由**并非一个独立的模块或 ML 驱动的意图识别系统，而是一套**多层次、多路复用的分层决策机制**，在用户消息进入系统后，在不同层级上完成路由决策：

- **命令路由（`gateway/run.py`）：** 字符串前缀匹配 + 命令注册表（`CommandDef` 数据类）。只有无法识别的命令才会分发给 LLM——一种有意的"负路由"模式（DFA）。
- **模型路由（`agent/smart_model_routing.py`）：** 逐轮关键词启发式规则（无 ML）。保守策略：任何代码块、URL 或复杂关键词（`debug`、`analyze`、`architecture` 等）触发强模型；否则使用配置的便宜模型。
- **消息分发路由（`gateway/delivery.py`）：** `DeliveryRouter` 解析 Cron 任务输出的目标——支持 `"origin"`、`"local"`、`"telegram"`、`"telegram:123456"` 格式，含三重键去重。
- **平台适配器路由（`gateway/platforms/`）：** 每个消息平台有专用适配器，启动时注册，支持热插拔。
- **子代理委托路由（`tools/delegate_tool.py`）：** 支持为子代理配置不同的 Provider/模型/工具集。
- **Hook 系统（`gateway/hooks.py`）：** 带通配符的事件路由（`"command:*"`），由 `HookRegistry` 支撑，含会话/项目/全局级作用域。

📄 **完整报告：** [意图路由.pdf](./ZH/意图路由.pdf)

---

### 3. 自进化机制

Hermes Agent 的自进化并非通过修改自身代码实现——而是通过**多层持久化知识管理系统形成闭环**：

- **Skills 层：** Agent 在复杂任务（5+ 次工具调用）、疑难错误或非平凡工作流后，通过 `skill_manage` 工具自主创建/补丁化 Skills。Skills 采用 YAML frontmatter + Markdown 正文格式，含版本控制、平台限制和子文件支持。
- **记忆层：** 三层混合持久化——文件系统存储（MEMORY.md/USER.md）、SQLite FTS5 跨会话搜索、Honcho 云服务语义用户建模。
- **RL 训练管道：** 三层解耦架构——轨迹收集（`trajectory.py`）→ 奖励计算（子类 `compute_reward` + `ToolContext`）→ 训练数据压缩（`trajectory_compressor.py`）。第一阶段生成 SFT 数据；第二阶段通过 vLLM 产生含精确 Token 级 logprobs 的真实 RL 训练数据。
- **反馈信号系统：** 自进化闭环的核心引擎。显式信号（工具调用次数、Token 限制）提供低延迟反馈；隐式信号（LLM 元认知、用户纠正）提供高适应性。核心悖论：谁来验证验证者？

📄 **完整报告：** [自进化机制.pdf](./ZH/自进化机制.pdf)

---

### 4. 工具系统

工具系统是 Hermes Agent 的执行基石，采用**插件化架构**，支持动态注册：

- **工具注册中心（`tools/registry.py`）：** 模块级单例，重复注册使用幂等覆盖（WARNING 而非异常），支持 MCP 动态重新注册。工具并发安全性在工具集级别分类。
- **工具集（`toolsets.py`）：** 通过 `tools` + `includes` 实现轻量级工具组合代数。递归解析含乐观循环检测（静默跳过），支持菱形依赖。
- **工具编排层（`model_tools.py`）：** 三层异步桥接处理同步/异步边界穿越。动态 Schema 修补移除对不可用工具的提及，防止名称幻觉。
- **工具调用解析器：** 支持 11+ 种模型特定格式（Hermes 微调、Llama、Qwen、Mistral、DeepSeek、GLM、Kimi K2 等），通过 `ToolCallParser` 注册表实现。
- **MCP 协议支持（`tools/mcp_tool.py`）：** 双向镜像——MCP Client（连接外部服务器）+ MCP Server（向 Claude Code/Cursor 暴露 Hermes）。线程隔离守护事件循环模式，含环境变量过滤、凭证清理和指数退避重连。
- **插件系统（`hermes_cli/plugins.py`）：** 三源发现（用户插件、项目插件、pip 包）。`PluginContext` 门面模式。生命周期钩子：`pre/post_tool_call`、`pre/post_llm_call`、`on_session_start/end`。
- **代码执行沙箱（`tools/code_execution_tool.py`）：** 多后端设计——Docker、Daytona、Modal、SSH、Singularity、本地开发。可用工具通过 `enabled_tools` 参数显式注入（能力空间隔离），而非从全局注册表查询。

📄 **完整报告：** [工具系统.pdf](./ZH/工具系统.pdf)

---

## 文档目录

| # | 主题 | English | 中文 |
|---|------|---------|------|
| 1 | 上下文管理 | [Context Management.pdf](./EN/Context%20Management.pdf) | [上下文管理.pdf](./ZH/上下文管理.pdf) |
| 2 | 意图路由 | [Intent Routing.pdf](./EN/Intent%20Routing.pdf) | [意图路由.pdf](./ZH/意图路由.pdf) |
| 3 | 自进化机制 | [Self-Evolution Mechanism.pdf](./EN/Self-Evolution%20Mechanism.pdf) | [自进化机制.pdf](./ZH/自进化机制.pdf) |
| 4 | 工具系统 | [Tool System.pdf](./EN/Tool%20System.pdf) | [工具系统.pdf](./ZH/工具系统.pdf) |

---

## 核心发现

- **零耦合工具自注册：** 工具通过 `registry.register()` 在导入时自注册，无需中央列表。MCP 动态重新注册使用幂等覆盖（WARNING 而非异常），支持工具列表的优雅变更。
- **确定性规则优先路由：** 意图路由完全基于规则（DFA、关键词启发式、字符串前缀匹配）——无 ML。这保证了确定性输出、微秒级延迟和成本可预测性，以可靠性换取灵活性。
- **"负路由"模式：** 只有无法识别的命令才分发给 LLM。这颠覆了典型的意图分类方法——LLM 是最后手段，而非第一响应者。
- **system_and_3 Prompt Caching：** 四个固定 `cache_control` 断点（系统 + 最近 3 条非系统消息）减少约 75% 的输入 Token 成本。最近的用户消息有意*不*被缓存，以支持并行 Agent 分叉场景。
- **迭代上下文压缩：** 每次压缩迭代基于前次摘要构建，而非从头重新摘要，避免多轮压缩带来的信号衰减（"摘要的摘要的摘要"）。
- **RL 管道完全解耦：** 奖励计算（`compute_reward`）是抽象方法——奖励函数可以是*任意*东西：代码正确性、用户满意度或自定义指标。`ToolContext` 提供真实验证（文件是否真实存在，测试是否通过），而非 LLM 自我评估。
- **通过显式参数化实现能力空间隔离：** 代码执行沙箱工具通过 `enabled_tools` 参数传递，而非从全局注册表查询。这防止了子代理通过执行任意代码逃逸父会话权限边界。
- **冻结快照模式：** 系统提示词在会话期间保持稳定，而 `memory` 工具响应反映最新状态——在不增加每次推理开销的前提下，解决了"一致性-新鲜度"的矛盾。
- **双层安全扫描：** `skills_guard.py`（纯 Python 正则，静态分析）+ `tirith_security.py`（外部 Rust 二进制，语义扫描）——均采用失败即放行策略（仅 WARNING，不阻断），防止安全工具本身成为单点故障。
- **反馈信号作为核心引擎：** 自进化由显式信号（5+ 次工具调用阈值、Token 计数）和隐式信号（LLM 元认知、用户纠正）共同驱动，形成类比 RL 奖励信号的"感知-反馈-调整"闭环。

