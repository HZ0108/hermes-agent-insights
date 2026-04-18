# Inside Hermes Agent: Architecture & Design Philosophy

> An in-depth source code analysis and architectural study of Hermes Agent by Nous Research.

---

For the Chinese version, please refer to [README_ZH.md](./README_ZH.md).

---

## Table of Contents

- [Inside Hermes Agent: Architecture & Design Philosophy](#inside-hermes-agent-architecture--design-philosophy)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Core Architectural Pillars](#core-architectural-pillars)
    - [1. Context Management](#1-context-management)
    - [2. Intent Routing](#2-intent-routing)
    - [3. Self-Evolution Mechanism](#3-self-evolution-mechanism)
    - [4. Tool System](#4-tool-system)
  - [Document Catalog](#document-catalog)
  - [Key Discoveries](#key-discoveries)

---

## Introduction

Hermes Agent is a self-improving AI agent developed by [Nous Research](https://github.com/nousresearch/hermes-agent), positioned as a "general-purpose, self-evolving, cross-platform" AI assistant. More than a simple API wrapper, it is a sophisticated multi-agent system designed to operate autonomously across local filesystems, remote environments, and messaging platforms (Telegram, Discord, Slack, WhatsApp, Signal).

By analyzing its source code, this project extracts the underlying Python structures to reveal how Hermes handles the hardest problems in AI agent development: **context window limits, long-term memory retention, dynamic tool registration, multi-tier intent routing, and model-level capability evolution.**

---

## Core Architectural Pillars

### 1. Context Management

Hermes Agent's context management system adopts a **layered decoupled** design philosophy, splitting context management into multiple independent modules with clearly defined responsibilities:

- **System Prompt Builder (`prompt_builder.py`):** Composes agent identity, platform hints, memory/skills guidance, project context files, and a two-level cached skills index (in-process LRU + disk snapshot).
- **Context References (`context_references.py`):** Supports `@file`, `@folder`, `@diff`, `@git`, `@url` syntax with strict token budget control (50% hard limit, 25% soft limit) and dual sandboxing (path boundary + sensitive path interception).
- **Context Compression (`context_compressor.py`):** Four-stage pipeline: tool output trimming → Head/Tail boundary protection → LLM structured summarization → tool pair integrity repair. Iterative compression preserves prior summaries to avoid signal degradation.
- **Prompt Caching (`prompt_caching.py`):** `system_and_3` strategy — fixes 4 `cache_control` breakpoints (system prompt + last 3 non-system messages), reducing input token costs by ~75%.
- **Message Format Adaptation (`anthropic_adapter.py`):** Bidirectional OpenAI ↔ Anthropic format conversion, with strict role alternation enforcement and OAuth compatibility handling (Claude Code masquerading).
- **Auxiliary Routing (`auxiliary_client.py`):** Unified provider resolution chain for compression/LLM calls — OpenRouter → Nous → Custom → Codex → Anthropic → Direct API-key.

📄 **Full Report:** [Context Management.pdf](./EN/Context%20Management.pdf)

---

### 2. Intent Routing

Hermes Agent's **intent routing** is not a standalone module or an ML-driven intent recognition system — it is a **multi-layered, multiplexed, hierarchical decision-making mechanism** that makes routing decisions at different levels:

- **Command Routing (`gateway/run.py`):** String prefix matching + command registry (`CommandDef` dataclass). Only unrecognized commands fall through to the LLM — a deliberate "negative routing" pattern (DFA).
- **Model Routing (`agent/smart_model_routing.py`):** Per-turn keyword heuristic rules (no ML). Conservative strategy: any code blocks, URLs, or complex keywords (`debug`, `analyze`, `architecture`, etc.) triggers the strong model; otherwise uses the configured cheap model.
- **Message Delivery Routing (`gateway/delivery.py`):** `DeliveryRouter` resolves targets for cron task outputs — supports `"origin"`, `"local"`, `"telegram"`, `"telegram:123456"` formats with three-key deduplication.
- **Platform Adapter Routing (`gateway/platforms/`):** Each messaging platform has a dedicated adapter registered at startup with hot-swap support.
- **Sub-Agent Delegation Routing (`tools/delegate_tool.py`):** Supports configuring different providers/models/toolsets for sub-agents.
- **Hook System (`gateway/hooks.py`):** Wildcard event routing (`"command:*"`) backed by a `HookRegistry`, with session/project/global scoping.

📄 **Full Report:** [Intent Routing.pdf](./EN/Intent%20Routing.pdf)

---

### 3. Self-Evolution Mechanism

Hermes Agent's self-evolution is not achieved by modifying its own code — it forms a **closed-loop multi-layer persistent knowledge management system**:

- **Skills Layer:** Agent autonomously creates/patches skills via `skill_manage` tool after complex tasks (5+ tool calls), tricky errors, or non-trivial workflows. Skills use YAML frontmatter + Markdown body format with version control, platform restrictions, and sub-file support.
- **Memory Layer:** Three-tier hybrid persistence — file storage (MEMORY.md/USER.md, SQLite FTS5 for cross-session search, Honcho cloud service for semantic user modeling).
- **RL Training Pipeline:** Complete three-layer decoupled architecture — trajectory collection (`trajectory.py`) → reward computation (subclass `compute_reward` + `ToolContext`) → training data compression (`trajectory_compressor.py`). Phase 1 generates SFT data; Phase 2 produces true RL training data with exact token-level logprobs via vLLM.
- **Feedback Signal System:** The core engine of the self-evolution loop. Explicit signals (tool call count, token limit) provide low-latency feedback; implicit signals (LLM self-assessment, user correction) offer high adaptability. The metacognitive dilemma: who verifies the verifier?

📄 **Full Report:** [Self-Evolution Mechanism.pdf](./EN/Self-Evolution%20Mechanism.pdf)

---

### 4. Tool System

The Tool System is Hermes Agent's execution backbone, built around a **plugin-based architecture** with dynamic registration:

- **Tool Registry (`tools/registry.py`):** Module-level singleton with idempotent override (WARNING on duplicate, not exception), enabling MCP dynamic re-registration. Tool concurrency safety is classified at the toolset level.
- **Toolsets (`toolsets.py`):** Lightweight tool composition algebra via `tools` + `includes`. Recursive resolution with optimistic cycle detection (silent skip). Supports diamond dependencies.
- **Tool Orchestration (`model_tools.py`):** Three-layer async bridging handles sync/async boundary crossing. Dynamic Schema patching removes mentions of unavailable tools to prevent name hallucination.
- **Tool Call Parsers:** Supports 11+ model-specific formats (Hermes custom fine-tuned, Llama, Qwen, Mistral, DeepSeek, GLM, Kimi K2, etc.) via `ToolCallParser` registry.
- **MCP Support (`tools/mcp_tool.py`):** Bidirectional mirror — MCP Client (connects to external servers) + MCP Server (exposes Hermes to Claude Code/Cursor). Thread-isolated daemon event loop pattern. Includes environment variable filtering, credential scrubbing, and exponential backoff reconnection.
- **Plugin System (`hermes_cli/plugins.py`):** Three-source discovery (user plugins, project plugins, pip packages). `PluginContext` facade pattern. Lifecycle hooks: `pre/post_tool_call`, `pre/post_llm_call`, `on_session_start/end`.
- **Code Execution Sandbox (`tools/code_execution_tool.py`):** Multi-backend design — Docker, Daytona, Modal, SSH, Singularity, local dev. Available tools are explicitly parameterized (capability space isolation), not queried from global registry.

📄 **Full Report:** [Tool System.pdf](./EN/Tool%20System.pdf)

---

## Document Catalog

| # | Topic | English | 中文 |
|---|-------|---------|------|
| 1 | Context Management | [Context Management.pdf](./EN/Context%20Management.pdf) | [上下文管理.pdf](./ZH/上下文管理.pdf) |
| 2 | Intent Routing | [Intent Routing.pdf](./EN/Intent%20Routing.pdf) | [意图路由.pdf](./ZH/意图路由.pdf) |
| 3 | Self-Evolution Mechanism | [Self-Evolution Mechanism.pdf](./EN/Self-Evolution%20Mechanism.pdf) | [自进化机制.pdf](./ZH/自进化机制.pdf) |
| 4 | Tool System | [Tool System.pdf](./EN/Tool%20System.pdf) | [工具系统.pdf](./ZH/工具系统.pdf) |

---

## Key Discoveries

- **Zero-Coupling Tool Registration:** Tools self-register at import time via `registry.register()`. No central list needed. MCP dynamic re-registration uses idempotent override with WARNING (not exception), enabling graceful tool list changes.
- **Deterministic Rule-First Routing:** Intent routing is entirely rule-based (DFA, keyword heuristics, string prefix matching) — no ML. This guarantees deterministic output, microsecond latency, and cost predictability, trading off flexibility for reliability.
- **"Negative Routing" Pattern:** Only unrecognized commands fall through to the LLM. This inverts the typical intent classification approach — the LLM is the last resort, not the first responder.
- **system_and_3 Prompt Caching:** Four fixed `cache_control` breakpoints (system + last 3 non-system messages) reduce input token costs by ~75%. The recent user message is intentionally *not* cached to support parallel agent forking scenarios.
- **Iterative Context Compression:** Rather than re-summarizing from scratch, each compression iteration builds on the prior summary, preventing signal degradation from multi-round compression ("summary of summary of summary").
- **RL Pipeline Decoupling:** Reward computation (`compute_reward`) is an abstract method — the reward function can be *anything*: code correctness, user satisfaction, custom metrics. `ToolContext` provides ground-truth verification (did files actually exist, did tests pass) rather than LLM self-assessment.
- **Capability Space Isolation via Explicit Parameterization:** Code execution sandbox tools are passed via `enabled_tools` parameter, not queried from global registry. This prevents sub-agents from escaping parent session permission boundaries through arbitrary code execution.
- **Frozen Snapshot Mode:** System prompt is stable throughout the session while `memory` tool responses reflect the latest state — solving the consistency-freshness contradiction without re-loading all memory on every inference.
- **Dual-Layer Security Scanning:** `skills_guard.py` (pure Python regex, static analysis) + `tirith_security.py` (external Rust binary, semantic scanning) — both fail-open (WARNING only, no blocking) to prevent the security tool itself from becoming a single point of failure.
- **Feedback Signals as the Core Engine:** Self-evolution is driven by explicit signals (5+ tool call threshold, token counts) and implicit signals (LLM metacognition, user corrections), forming a "perceive-feedback-adjust" closed loop analogous to RL reward signals.

