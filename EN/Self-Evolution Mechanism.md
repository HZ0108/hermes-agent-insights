# Self-Evolution Mechanism

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Self-Evolution Architecture Overview](#2-self-evolution-architecture-overview)
   - [Self-Evolution Feedback Loop](#self-evolution-feedback-loop)
3. [Core Self-Evolution Mechanisms in Detail](#3-core-self-evolution-mechanisms-in-detail)
   - [3.1 Skill Creation & Self-Maintenance](#31-skill-creation--self-maintenance)
   - [3.2 Memory Persistence System](#32-memory-persistence-system)
   - [3.3 Cross-Session Retrieval & Pattern Recognition](#33-cross-session-retrieval--pattern-recognition)
   - [3.4 Context Compression & Information Protection](#34-context-compression--information-protection)
   - [3.5 RL Training Data Pipeline](#35-rl-training-data-pipeline)
4. [Auxiliary Evolution Subsystems](#4-auxiliary-evolution-subsystems)
   - [4.1 User Modeling (Honcho)](#41-user-modeling-honcho)
   - [4.2 Multi-Instance Profile Isolation](#42-multi-instance-profile-isolation)
   - [4.3 MCP Dynamic Tool Discovery](#43-mcp-dynamic-tool-discovery)
5. [Tool Layer Self-Evolution Support](#5-tool-layer-self-evolution-support)
6. [Trajectory Engineering: Data-Driven Evolution](#6-trajectory-engineering-data-driven-evolution)
7. [Evolution Mechanism Relationship Diagram](#7-evolution-mechanism-relationship-diagram)
8. [Technical Implementation Details](#8-technical-implementation-details)
   - [8.1 Skill Activation Mechanism](#81-skill-activation-mechanism)
   - [8.2 Memory Injection Strategy](#82-memory-injection-strategy)
   - [8.3 Security Mechanisms](#83-security-mechanisms)
   - [8.4 Platform Adaptation Layer](#84-platform-adaptation-layer)
9. [Summary & Evaluation](#9-summary--evaluation)
   - [9.1 Core Design Philosophy](#91-core-design-philosophy)
   - [9.2 Evolution Mechanism Classification](#92-evolution-mechanism-classification)
   - [9.3 Innovations](#93-innovations)
   - [9.4 Feedback Signal System Full-Scope Analysis](#94-feedback-signal-system-full-scope-analysis)
   - [9.5 Limitations](#95-limitations)
10. [Reference File Index](#reference-file-index)

---

## 1. Project Overview

**Hermes Agent** is a self-improving AI agent developed by [Nous Research](https://github.com/nousresearch/hermes-agent), positioned as a "general-purpose, self-evolving, cross-platform" AI assistant. Its core philosophy is:

> *"Create Skills that learn from experience, continuously improve through use, and run anywhere."*

### Key Features

| Feature | Description |
|---------|-------------|
| **==Built-in Learning Loop==** | **==Discover and solidify knowledge from complex tasks, error corrections, and non-trivial workflows==** |
| Memory System | File-based persistent memory (MEMORY.md / USER.md) + Honcho user modeling |
| Cross-Session Retrieval | FTS5 full-text search + LLM summarization compression, supporting context reuse across sessions |
| Skill Management | Agent can autonomously create, edit, and patch skills, with runtime self-maintenance |
| RL Training Support | Trajectory saving, compression, and Tinker-Atropos RL environment integration |
| Multi-Platform Gateway | Telegram, Discord, Slack, WhatsApp, Signal messaging platforms |
| MCP Integration | Dynamic discovery and invocation of external tools provided by MCP servers |

---

## 2. Self-Evolution Architecture Overview

Hermes Agent's self-evolution is not achieved by modifying its own code, but through a **multi-layer persistent knowledge management system** that forms a closed loop:

```
┌─────────────────────────────────────────────────────────┐
│                     User Interaction Layer               │
│    CLI / Telegram / Discord / Slack / VS Code / ACP     │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│                  Agent Core Loop                        │
│   run_agent.py / agent/run.py                           │
│      - Intent Understanding → Tool Invocation → Response Generation → Feedback Collection │
└────────────────────┬────────────────────────────────────┘
                     │
    ┌─────────────────┼─────────────────────────────────┐
    │                 │                                  │
    ▼                 ▼                                  ▼
┌────────┐      ┌──────────┐      ┌─────────────────┐
│ Skills │      │  Memory  │      │ Trajectory/RL   │
│  Layer │      │   Layer  │      │     Layer       │
│        │      │          │      │                 │
│Create  │      │MEMORY.md │      │ Trajectory Save │
│ Patch  │      │ USER.md  │      │ Compression     │
│ Skills │      │ Honcho   │      │ Atropos RL      │
│ Skills │      │ Session  │      │                 │
│  Hub   │      │ Search   │      │                 │
└────────┘      └──────────┘      └─────────────────┘
```

### Self-Evolution Feedback Loop

```
Execute Task
  ↓
Detect Complex Task / Error / New Workflow
  ↓
skill_manage Create or Patch Skill ──→ Persist Skill to ~/.hermes/skills/
  ↓
Update Memory Files (MEMORY.md / USER.md)
  ↓
Trajectory Saving (batch_runner.py / trajectory.py)
  ↓
RL Training Data Compression (trajectory_compressor.py)
  ↓
Next Session: Retrieve Historical Skills / Memory / Sessions → Reuse Experience
```


> [!NOTE]
> **Feedback signals are the key to the self-evolution closed loop:**
> - The agent's self-evolution is essentially a "perceive-feedback-adjust" closed-loop process. Whether it's skill creation, memory updates, or trajectory compression, all rely on clear feedback signals to drive the process: Was the task successful? Is the user satisfied? What improvements were made compared to historical solutions?
> - This is highly aligned with the design philosophy of Reinforcement Learning (RL): the external environment provides rewards/penalties, and the agent adjusts its strategy accordingly. The difference is that the agent's feedback signal sources are more diverse—not only from task success rates, but also from user-explicit corrections, tool call exceptions, context compression loss evaluations, and other multi-dimensional information. How to establish a unified feedback signal measurement system is one of the core challenges in self-evolving agent design.


---

## 3. Core Self-Evolution Mechanisms in Detail

### 3.1 Skill Creation & Self-Maintenance

#### 3.1.1 Skill Management Tool

**File**: `tools/skill_manager_tool.py`

The agent performs complete CRUD operations on skills via the `skill_manage` tool:

```python
# Supported action parameters
CREATE   # Create new skill (extracted from successful experience)
EDIT     # Edit full content of existing skill
PATCH    # Small-scale patch (quick fix for errors/omissions)
DELETE   # Delete deprecated skill
WRITE_FILE   # Create skill sub-files (references, templates, assets)
REMOVE_FILE  # Delete skill sub-files
```

#### 3.1.2 Feedback Signals & Trigger Conditions (Design Philosophy)

> [!NOTE]
> **Core Design Insight: Implicit Signal Dependency is a Double-Edged Sword**
>
> Hermes Agent's skill triggering mechanism is built on a **hybrid feedback signal system**, whose core contradiction is: **the most valuable feedback signals (whether a task is valuable, whether it's worth solidifying) are precisely the hardest to quantify.**

**Distribution of Quantitative vs. Qualitative Signals:**

| Trigger Condition | Signal Type | Verifiability | Advantage | Risk |
|----------|---------|---------|------|------|
| Complex task (**5+ tool calls**) | Explicit quantitative | High (precise counting) | Reproducible, unambiguous | Cannot distinguish "complex wasted effort" from "complex but valuable" |
| **Tricky error** overcome | Implicit semantic | Low (depends on LLM judgment) | Can capture truly valuable experience | May over-create skills (create a skill for every error) |
| **Non-trivial workflow** discovered | Implicit semantic | Low | Can capture unpredictable patterns | Judgment is subjective, relies on model's own metacognition |
| Method works **after user correction** | Implicit social | Medium (requires understanding correction intent) | User participation reduces error cost | Over-reliance on user intervention, weakens autonomy |

**The 5+ Call Threshold is a Reasonable Engineering Approximation**
- "5" is an empirical value, essentially a **trade-off between recall rate (not missing good skills) and precision rate (not introducing noisy skills)**
- A more elegant design might introduce **task value evaluation** as a second signal, forming joint trigger conditions with call count

> [!IMPORTANT]
> **Insight: CREATE Trigger is Essentially "Experience Distillation"**
> - The agent not only executes tasks but also continuously evaluates "is this experience generalizable" during execution
> - The 5+ call threshold reduces misjudgment cost: if a workflow repeats multiple times and uses 5+ tools each time, it's likely worth solidifying
> - This is essentially an **exploration-exploitation trade-off in Online Learning**: the agent decides while executing whether to record the current experience into long-term memory


**Skill Creation Trigger Conditions:**

| Trigger Scenario | Example |
|----------|------|
| **Complex task (5+ tool calls) successfully completed** | Completed a refactoring task spanning multiple files |
| **Tricky error overcome** | Discovered and worked around a non-obvious bug |
| **Method works after user correction** | Successfully completed the task after user pointed out the right direction |
| **Non-trivial workflow discovered** | Found a better CI/CD pipeline |

#### 3.1.3 Maintenance Trigger Conditions

The agent is explicitly instructed to **automatically patch skills** in the following situations:

| Trigger Scenario | Maintenance Action |
|----------|----------|
| Encountered problem when using a skill | Immediately PATCH, don't wait to be asked |
| Instructions are outdated or incomplete | Fill in missing steps and known pitfalls |
| OS-specific failure | Add platform-related patch |
| New pitfall discovered | Add warning in SKILL.md |

#### 3.1.4 Skill File Format

Skills use a **YAML frontmatter + Markdown body** structured format:

```yaml
---
name: git-interactive-rebase
description: Use interactive rebase to organize commit history
version: 1.2.0
platforms: [macos, linux]
tags: [git, vcs, history]
author: hermes-agent  # tagged when agent self-creates
auto_updated: true     # marked as agent self-maintained
---

# Git Interactive Rebase Skill

## Applicable Scenarios
Use this when you need to organize commit history, merge WIP commits, or modify commit messages.

## Usage Steps
1. Use `git log --oneline -n 20` to view recent commits
2. Determine the number n of commits to modify
3. Execute `git rebase -i HEAD~n`
4. Select operations in the editor (pick/reword/squash/fixup/drop)

## Common Pitfalls
- ⚠️ Do not rebase commits that have been pushed (unless you are sure there is no collaboration)
- ⚠️ When handling conflicts, resolve them in order, do not skip any
```

> [!NOTE]
> Progressive disclosure and lazy-loading flow as proposed by Anthropic

#### 3.1.5 Skill Directory Structure

```
~/.hermes/skills/
├── git-interactive-rebase/
│   ├── SKILL.md              # Main skill file
│   ├── references/           # Reference documents
│   │   └── git-rebase-demo.md
│   └── templates/            # Output templates
│       └── commit-message-template.txt
├── docker-debugging/
│   └── SKILL.md
└── swe-expert/
    └── SKILL.md
```

#### 3.1.6 Skill Registration to System Prompt

**File**: `agent/prompt_builder.py`

Skills are **automatically loaded** into the agent's system prompt via `build_system_prompt()` in `prompt_builder.py`:

```python
def build_system_prompt(identity, hints, skills, env_vars, soyl):
    # 1. Load identity description (SOUL.md)
    # 2. Apply hints — includes skill usage guidance
    # 3. Inject active skill list and their descriptions
    # 4. Expose environment variables
    # 5. Output complete system prompt
```


---

### 3.2 Memory Persistence System

Hermes Agent builds a three-layer memory system to achieve cross-session knowledge persistence.

#### 3.2.0 Storage Medium Layered Design

> [!NOTE]
> **Storage medium is not chosen arbitrarily, but weighed by "access frequency × consistency requirement × human editability".**

```
┌──────────────────────────────────────────────────────────┐
│     "How fast do I need to write? How strong consistency?  │
│                Can humans edit it?"                       │
└──────────────────────────────────────────────────────────┘
                          │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
    High-freq write +    Low-freq write +  External service
    structured query     human-editable
          │                │                │
          ▼                ▼                ▼
     SQLite + FTS5      File system        Honcho
     (session state)    (long-term memory)  (semantic memory)


┌─────────────────────────────────────────────────────────────┐
│   Why SQLite for sessions, file system for long-term memory? │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Session Memory (SQLite)          Long-term Memory (File)   │
│  ─────────────────────            ───────────────────────    │
│  • Real-time write per message    • Updated rarely          │
│  • Needs FTS5 full-text search    • Humans can edit directly │
│  • Multi-platform concurrent     • Can git diff/rollback   │
│    writes (WAL)                                             │
│  • Messages compressed before     • Direct access by tools  │
│    storage                          (file_tools etc.)        │
│                                                               │
│  Honcho is the exception — stored in external service,      │
│  opaque but naturally syncs across devices                   │
└─────────────────────────────────────────────────────────────┘
```

**Storage Medium Comparison for Three Memory Layers:**

| Memory Layer | Storage Medium | Write Frequency | Query Mode | Human-Editable | Version Control |
|--------|---------|---------|---------|---------|---------|
| Session State | SQLite + FTS5 | Real-time per message | Structured + full-text search | ❌ | ❌ |
| MEMORY/USER.md | File system (`~/.hermes/memories/`) | Very low (on-demand) | Full read | ✅ Direct edit | ✅ Git |
| Honcho | External service (HTTP API) | On-demand | Semantic search | ❌ | ❌ |

**Potential Risks of Cross-Layer Consistency (not covered by current framework):**

```
MEMORY.md (file system)              sessions.db (SQLite)
        │                                  │
        ▼                                  ▼
  Manual/latency update             Auto compression then write
        │                                  │
        └──────── Version drift may develop ────────┘
                   After Honcho writes, USER.md doesn't auto-sync;
                   and vice versa. Over long runs, the agent's
                   understanding of the user may "drift" in version.
```

#### 3.2.1 Layer 1: File Memory (MEMORY.md / USER.md)

**File**: `tools/memory_tool.py`

**MEMORY.md** — Agent's private notes:
- Environmental facts (project structure, tool habits, system differences)
- Project conventions (code style, naming conventions)
- Tool defects and known issues
- Configuration information (key configurations to avoid detours)

**USER.md** — Agent's understanding of the user:
- User roles and responsibilities
- Preferences and communication style
- Expectations and requirement patterns
- Collaboration habits


#### 3.2.2 Layer 2: Session State (SQLite + FTS5)

**File**: `hermes_state.py`

```python
# SQLite Schema
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,         # Unique session identifier (UUID)
    source TEXT,                 # 'cli' | 'telegram' | 'discord' etc., supports cross-platform retrieval
    model TEXT,                  # Model name used in this session (e.g., 'claude-sonnet-4-6')
    created_at INTEGER,           # Unix timestamp (seconds), session creation time
    updated_at INTEGER,           # Unix timestamp (seconds), last message time, used for LRU cleanup
)

CREATE VIRTUAL TABLE sessions_fts USING fts5(
    # ── FTS5 full-text index columns (content actually stored in sessions table) ──
    content,                     # Message body (post-context-compression version)
    role,                        # Message role: 'user' | 'assistant' | 'tool'
    session_id,                  # Foreign key to sessions.id, supports grouped retrieval by session
    token_count,                 # Token count for this message, used for sorting and pagination
    # ── FTS5 content sync config ──────────────────────────────────────
    content='sessions',          # Points to sessions table (not independently stored), avoids data redundancy
    content_rowid='rowid',       # Associates original records via rowid, ensures index and table stay in sync
);
```

> [!NOTE]
> `content_rowid='rowid'` is FTS5's "content sync" design: the index itself doesn't store content, but joins back to the sessions table in real-time via rowid. This design ensures FTS5 index is always consistent with raw data while saving index storage overhead — in read-heavy concurrent scenarios, the cost of one extra JOIN is negligible.

**Characteristics:**
- WAL mode: concurrent reads + single write
- FTS5 full-text index: millisecond-level cross-session search
- Session source distinguished by `source` tag
- Per-message token counting

#### 3.2.3 Layer 3: Skills System (Persistent Knowledge Modules)

**File**: `tools/skills_tool.py` / `tools/skill_manager_tool.py`

Skills as **executable, patchable persistent knowledge modules** go a step further than MEMORY.md:

- Have structured metadata (name, version, platform, tags)
- Can contain sub-files (references, templates, assets)
- Support version control (explicit version field)
- Have active/inactive status (can be toggled in config)
- Have security scanning mechanism (pre-install validation)


#### 3.2.4 Feedback Signal Design Philosophy

> [!IMPORTANT]
> **Core Design Insight: The Feedback Signal of Memory is Essentially "Information Lifespan Expectation"**
>
> Hermes Agent's memory system is not passive storage, but an **active information lifecycle management system**. Different memory layers correspond to different feedback loop frequencies and information fidelity requirements.

**Three-Layer Memory Signal Feedback Characteristics Comparison:**

| Memory Layer | Feedback Signal Source | Update Frequency | Signal Type | Consistency Mechanism |
|--------|------------|---------|---------|-----------|
| MEMORY.md | Agent self-assessment (LLM judges "is this worth remembering") | Very low (cross-month) | Implicit semantic | Frozen snapshot mode |
| SQLite sessions | External retrieval queries trigger | Medium (weekly) | Explicit query + FTS5 | WAL concurrency control |
| Skills system | Skill execution results + user feedback | Low (weekly) | Implicit + explicit mixed | Version field + snapshot cache |


---

**Design Philosophy:**

| Design Decision | Reason |
| ----------------------- | ------------------------ |
| Frozen Snapshot Mode | Stable system prompt, but tool responses show latest state |
| Character limit (not token limit) | Model-agnostic (doesn't depend on specific model's tokenizer) |
| Injection attack scanning | Prevent user-injected malicious content from manipulating the agent |

**The Design Philosophy of "Frozen Snapshot" Mode**
- System prompt stays stable during the session, but each `memory` tool response reflects the latest state
- This solves the **consistency-freshness contradiction** in LLM reasoning: reasoning needs stable context, but reality is continuously changing
- Insight: this is actually a kind of **optimistic caching + delayed invalidation** strategy — assuming the snapshot is good enough in the short term, using tool calls to get incremental updates
- Comparison: another approach is to reload all memory every time, but that would cause unnecessary context overhead for every inference

**The Deep Reason for Character Limits (not token limits)**
- Tokenization is model-dependent, different models have different tokenizers
- But the deeper design intent is: **let memory content be independent of the model, becoming true "persistent knowledge"**
- This allows memory to migrate between different models, and makes memory management tools more stable

**The Reverse Signal Value of Injection Attack Scanning**
- The scan itself is also a "negative feedback" signal: any blocked content is a manipulation signal the user tried to inject
- This passive detection signal can conversely help identify **high-risk user interaction patterns** (is someone persistently trying to inject?)

---

### 3.3 Cross-Session Retrieval & Pattern Recognition

**File**: `tools/session_search_tool.py`

This is the key mechanism for Hermes Agent to achieve "experience reuse".

#### Workflow

```
User query "How to handle concurrency in Python?"
       ↓
FTS5 full-text search → Find relevant sessions (top-3 unique sessions)
       ↓
Group by session, get relevant messages in session
       ↓
LLM summarization (cheap fast model)
       ↓
Return: Focused summary (not raw transcript)
       ↓
Agent integrates historical experience into current task
```

#### Synergy with Memory System

| Retrieval Target | Tool | Storage Location |
|----------|------|----------|
| Historical conversation experience | `session_search` | SQLite FTS5 |
| Agent's own notes | `memory` | MEMORY.md |
| User preferences | `memory` + `honcho` | USER.md + Honcho |
| Structured skills | `skills` / `skill_manage` | ~/.hermes/skills/ |

---

### 3.4 Context Compression & Information Protection

**File**: `agent/context_compressor.py`

#### Compression Strategy

```python
# Protected areas (never compressed):
PROTECTED_HEAD = ["system", "first_human", "first_gpt", "first_tool"]

# Compression area:
# - Middle messages exceeding token_budget
# - Compressed via LLM summarization
# - Preserve enough context for tool calls

# Tail protection:
# - Last ~20K tokens kept in original (latest context matters most)
```

#### 3.4.1 Feedback Signal Design for Context Compression

> [!IMPORTANT]
> **Core Design Insight: The Feedback Signal of Compression is "Information Compressibility" not "Information Importance"**
>
> The core assumption of context compression is: **after summarization compression, historical middle-round information can still retain enough "signal" to drive correct reasoning**. This is a bold empirical assumption, not a theoretical guarantee.

**Signal Value Analysis of Three-Layer Protection:**

| Protection Layer | Protected Content | Signal Source | Design Logic |
| -------- | ----------------------- | --------- | -------------------- |
| **Head protection** | system + first exchange | Structural position | Reasoning foundation framework; once destroyed, entire session is meaningless |
| **Tail protection** | Recent ~20K tokens | Dynamic token budget | Latest context directly affects current reasoning quality |
| **Middle compression** | Historical interactions | LLM summary quality | "Semantic summary" of historical experience retains patterns, not details |

**The Design Value of Iterative Summarization (`self._previous_summary`)** -> ==Segmented Progressive Incremental Compression Method==
- First compression: generate summary from scratch
- Second+ compression: update based on previous summary, instead of re-summarizing
- This is consistent with the idea of **differential compression**: incremental changes only need to record incremental information
- Avoids signal degradation from over-summarization of session history after multiple compressions ("summary of summary of summary")

**The Design Intent of Structured Summary Template (Goal/Progress/Decisions/Files/Next Steps)**
- Inspired by **Pi-mono and OpenCode** experience
- Each field corresponds to one type of information: goal (direction), constraints (preferences), progress (state), decisions (path), files (resources), next steps (plan)
- Insight: this six-tuple actually encodes the complete **MDP state** of agent task execution (MDP = Markov Decision Process)

```
Message list
    ↓
Identify PROTECTED_HEAD (system-level messages)
    ↓
Identify PROTECTED_TAIL (recent ~20K tokens)
    ↓
Middle area → LLM summarization → Single summary message replacement
    ↓
Verify compressed token_count ≤ token_budget
    ↓
If still over limit, recursive compression (summarize again)
```

---

### 3.5 RL Training Data Pipeline

Hermes Agent has a built-in complete RL training data pipeline that **converts agent experience into trainable RL data**.

**Core Design Insight: Hermes Agent's RL pipeline is a "validation-data-training" three-layer decoupled architecture**, **Signal Flow of Three-Layer Decoupled Architecture:**

```
Layer 1: Trajectory Collection (HermesAgentLoop)
  → Output: AgentResult(messages, turns_used, tool_errors, reasoning)
  → Feedback signal: Tool call sequence + error logs (structured intermediate process data)

Layer 2: Reward Computation (subclass compute_reward + ToolContext)
  → Input: AgentResult + ToolContext (complete tool access)
  → Output: float reward (usually 0.0-1.0, but any float valid)
  → Feedback signal: defined by subclass (reward function can be anything: code correctness, user satisfaction, or custom metrics) — this is the most critical signal source

Layer 3: Training Data Compression (TrajectoryCompressor)
  → Input: Raw trajectory + reward
  → Output: Compressed trajectory (ShareGPT format)
  → Compliance constraint: training framework context window upper limit (15,250 tokens) — exceeding this won't fit in the trainer, hard requirement
```


> [!NOTE]
> **Note: Layer 3 does not use ContextCompressor**
>
> Trajectory compression is not "in-session context compression"; they are two completely independent modules:
> - `agent/context_compressor.py`: triggered in real-time during session, operates on `{role/content}` format, serves **reasoning**
> - `trajectory_compressor.py`: offline batch processing after session ends, operates on **ShareGPT JSONL**, serves **training data generation**
>
> The only commonality is both call `call_llm()` for LLM summarization, but even the prompt templates differ — **==the former outputs a structured six-tuple, the latter outputs natural language description==**.


---

> [!NOTE]
> The innovation of the RL pipeline is the **complete decoupling of reward function from training framework**: `compute_reward()` is an abstract method implemented by subclasses, meaning **the reward function can be anything: code correctness, user satisfaction, or custom metrics**.

**ToolContext as the "Universal Interface" for Reward Functions**

- `compute_reward()` gets complete tool access (terminal, file, web, browser...)
- This means reward functions can **directly execute real-world verification**: not just seeing what the agent says, but verifying it did the right thing
- Example: check if files were actually created, commands actually succeeded, tests actually passed
- Insight: this is essentially **Ground Truth verification**, not LLM self-assessment — the reward signal comes from the objective world, not the agent's self-perception

---

**Signal Quality Difference Between Two-Phase (Phase 1 / Phase 2) Design**

| Phase | Server Type | Token Tracking | Applicable Scenario | Training Signal Quality |
|------|-----------|----------|---------|-----------|
| Phase 1 | OpenAI-compatible | None (placeholder tokens) | SFT data generation, validation tests | Cannot be used for RL (missing logprobs) |
| Phase 2 | vLLM/SGLang ManagedServer | Exact token IDs + logprobs | RL training (GRPO/PPO) | Complete (usable for advantage estimation) |

- Phase 1 trajectory data has messages but no precise token-level signals, cannot compute advantage estimation
- Phase 2 is true RL training mode, but requires vLLM server infrastructure
- This phased design allows **first collect data (Phase 1), then upgrade to training (Phase 2)**, reducing cold-start threshold

#### 3.5.1 Trajectory Saving

**File**: `agent/trajectory.py` / `trajectory_compressor.py`

A Trajectory is the complete sequence of agent-environment interactions:

```json
{
  "session_id": "xxx",
  "model": "claude-sonnet-4-6",
  "trajectory": [
    {"role": "user", "content": "Help me refactor this module..."},
    {"role": "assistant", "content": "...", "tool_calls": [...]},
    {"role": "tool", "tool_call_id": "...", "content": "..."},
    {"role": "assistant", "content": "Okay, I'll start refactoring..."},
  ],
  "success": true,
  "tool_count": 7,
  "created_at": 1712000000
}
```

#### 3.5.2 Trajectory Compression — "Lossy Compression" of Training Signals

> [!IMPORTANT]
> **Core Design Insight: For training trajectory compression, quality depends on understanding "where are the training signals"**
>
> Trajectory compression differs from context compression: the latter serves reasoning quality, the former serves **training signal quality**. This means when judging "what to compress and what to keep", the criterion is not "is this important for the current task", but "is this important for model learning" — these can be vastly different.

**Three-Priority Model for Training Signals:**

| Priority | Content | Reason |
|--------|------|------|
| **P0 Must Retain** | system prompt + first human + first gpt + first tool | "Anchors" for behavioral pattern learning,决定了策略的初始方向 |
| **P1 Must Retain** | last 4 turns | Final state and conclusion directly affect reward; tail contains the clearest causal signals |
| **P2 Can Compress** | All middle turns | As long as the summary retains key decision paths, intermediate process details can be compressed |

**Why Summary Strategy is "Descriptive Summary" not "Reasoning Process Extraction"?**
- Summary prompt requires: describe what the assistant **did** (tool calls, searches, file operations), not what it **thought**
- Reason: in training data, the model's tool call sequence is the behavior to be learned, not the reasoning chain
- This contrasts with `reasoning_per_turn` design: reasoning content is separately extracted for display (wandb), but not injected into training data

**The Meta-Value of Compression Metrics: Signals Themselves are Data**
- `TrajectoryMetrics` records: raw tokens, post-compression tokens, compression ratio, API call count, error count
- These metrics are not just "logs", but **implicit feedback signals for compression quality**
- Example: `still_over_limit: true` means compression wasn't aggressive enough, need to adjust `target_max_tokens` next time
- `compression_ratio` distribution (min/max/median) is the basis for judging whether compression strategy is stable


#### 3.5.3 Atropos RL Environment Integration

**File**: `environments/hermes_base_env.py` / `environments/agent_loop.py` / `tools/rl_training_tool.py`

Integration with the Tinker-Atropos RL framework:

```python
# Two run modes:
# Phase 1: OpenAI-compatible server
# Phase 2: vLLM ManagedServer (GPU-accelerated inference)

# Reward function design:
class ToolContext:
    """Provides tool access for each rollout, used for custom reward computation"""
    get_tool_calls()      # Get tool call sequence
    get_rewards()         # Compute reward signal
    get_trajectory()      # Get complete trajectory
```

**Training Environments:**

| Environment | Purpose |
|------|------|
| `HermesSweEnv` | SWE-bench style software engineering tasks |
| `TerminalBench2EvalEnv` | Terminal benchmark evaluation |
| `TerminalTestEnv` | Stack verification tests |

---

## 4. Auxiliary Evolution Subsystems

### 4.1 User Modeling (Honcho)

**File**: `tools/honcho_tools.py` / `honcho_integration/session.py`

> [!IMPORTANT]
> **Core Design Insight: Honcho's "Dialectic Understanding" is Essentially Distilling User Feedback Signals into Structured Priors**
>
> Honcho doesn't store what the user "said", but what it "understood" — this is a distillation process from raw signals to abstract representations (or a process of **latent knowledge mining** from interaction history). This design solves a fundamental problem: **users won't proactively tell you who they are, they'll only tell you what they want.**

Honcho is Nous Research's AI-native user modeling system (specifically stores knowledge "about the user" such as preferences, styles, habits), providing **dialectic user understanding** that goes beyond simple key-value storage:

| Tool | Function |
|------|----------|
| `honcho_profile` | Retrieve/update user refined profile (peer card) |
| `honcho_search` | Semantic search of stored user context |
| `honcho_context` | LLM-driven dialectic Q&A (understanding user intent) |
| `honcho_conclude` | Write conclusions back to Honcho long-term memory |

> [!NOTE]
> **Honcho's Storage Medium: External Service (HTTP API)**
>
> Unlike SQLite and file system, Honcho's data is stored on Nous Research's Honcho server; locally only a message cache is maintained (`honcho_integration/session.py`), synchronized with the service via HTTP API. This design **makes user profiles naturally shared across devices and sessions** — switch to another machine and you can still read the same profile.
>
> The cost is: dependence on network availability, server-side data not directly editable locally, and consistency between Honcho and local MEMORY.md / USER.md needs to be manually synchronized via `honcho_conclude`.


**User Modeling Flow:**

```
Every session interaction
    ↓
Collect key information (preferences, habits, feedback)
    ↓
honcho_conclude: Write back to Honcho
    ↓
Next session honcho_profile: Retrieve user profile
    ↓
honcho_search: Semantic search historical context
    ↓
Agent adjusts response strategy based on user profile
```

### 4.2 Multi-Instance Profile Isolation

**File**: `hermes_cli/profiles.py`

Hermes Agent supports multi-instance Profiles, completely isolated operating environments:

```
~/.hermes/
├── config.yaml              # default profile
├── .env
├── memories/
├── skills/
├── sessions.db
└── profiles/
    ├── work/                 # work scenario profile
    │   ├── config.yaml
    │   ├── .env
    │   ├── memories/
    │   ├── skills/
    │   └── sessions.db
    ├── research/             # research scenario profile
    │   └── ...
    └── personal/
        └── ...
```

**Isolated Content:**
- Configuration files (config.yaml)
- API keys (.env)
- Memory files (memories/)
- Skill collections (skills/)
- Session history (sessions.db)
- Gateway configuration (gateway/)
- Scheduled tasks (cron/)


> [!NOTE]
> The design of Honcho in Hermes is user (person)-oriented, not profile-oriented:
>   - Profile is "work scenario isolation"
>   - Honcho is "same person's cognitive sharing"
>   So if you talk about a lot of work preferences in the work Profile, after switching to personal Profile, Honcho still remembers them — shared across all Profiles

### 4.3 MCP Dynamic Tool Discovery

**File**: `tools/mcp_tool.py`

MCP (Model Context Protocol) support enables the agent to **dynamically expand tool capabilities**:

```python
# MCP tool registration flow
mcp.discover_servers()     # Discover available MCP servers
    ↓
mcp.list_tools()          # Get tool list
    ↓
mcp.call_tool(name, args) # Invoke tool
    ↓
notifications/tools/list_changed  # Monitor tool changes
    ↓
Dynamically update agent toolset
```

---

## 5. Tool Layer Self-Evolution Support

### Tool Registry

**File**: `tools/registry.py`

A unified tool registry supporting tool discovery, registration, permission control, and distribution:

```python
# ── Tool registry ─────────────────────────────────────────────
# Each tool module calls register() on import, declaring itself to the global registry
# model_tools.py only queries the registry, no longer maintains its own tool list
# ──────────────────────────────────────────────────────────────────────

registry.register(
    name="skill_manage",          # Unique identifier for the tool, dispatch looks up by name
    toolset="skills",             # Toolset the tool belongs to, used for batch enable/disable

    # ── OpenAI tool_calls format schema ──────────────────────────────
    # Tells LLM: what this tool is called, what it does, what parameters it accepts
    schema={
        "name": "skill_manage",   # Must match name field inside schema
        "description": "Create, edit, or patch skills...",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["create", "edit", "patch", "delete"]
                    # action determines operation type:
                    #   create   — create new skill from successful experience
                    #   edit     — complete rewrite of skill content
                    #   patch    — quick patch (for specific errors/omissions)
                    #   delete   — delete deprecated skill
                },
                ...
            }
        }
    },

    # ── Actual execution logic of the tool ────────────────────────────────────────────
    # The real work function, called when registry.dispatch(name) is invoked
    # Can be a regular function or an async function (marked by is_async)
    handler=lambda args, **kw: skill_manager_tool(...),

    # ── Availability check function ─────────────────────────────────────────────
    # dispatch calls check_fn() first, only exposes tool to LLM if it returns True
    # Returns False = tool is invisible to LLM (not an error)
    # check_fn can check: environment variables, dependencies installed, API key configured etc.
    # skill_manage requires HERMES_SKILLS_DIR to exist, otherwise skills can't be created
    check_fn=check_requirements,

    # ── Required environment variables list ─────────────────────────────────────────────
    # Not a hard constraint (check_fn is), but metadata
    # Used for hints/docs: tells which environment variables this tool depends on
    # MCP dynamic discovery will check these, ensuring tools can run properly
    requires_env=["HERMES_SKILLS_DIR"],
)
```

### Tool Classification (Toolsets)

**File**: `toolsets.py`

```python
TOOLSETS = {
    "web":        ["web_tools"],
    "search":     ["web_tools"],
    "vision":     ["vision_tool"],
    "image_gen":  ["image_generation_tool"],
    "terminal":   ["terminal_tool", "environments"],
    "skills":     ["skills_tool", "skill_manager_tool"],
    "browser":    ["browser_tool"],
    "cronjob":    ["cron_tool"],
    "messaging":  ["messaging_tools"],
    "rl":         ["rl_training_tool", "trajectory_compressor"],
    "file":       ["file_tools"],
    "tts":        ["tts_tool"],
    "todo":       ["todo_tool"],
    "memory":     ["memory_tool"],
    "session_search": ["session_search_tool"],
    "clarify":    ["clarification_tool"],
    "code_execution": ["code_execution_tool"],
    "delegation": ["delegate_tool"],
    "honcho":     ["honcho_tools"],
    "homeassistant": ["homeassistant_tool"],
}
```

---

## 6. Trajectory Engineering: Data-Driven Evolution

### Complete Trajectory Lifecycle

```
Step 1: Session Execution
        run_agent.py / batch_runner.py
             ↓
Step 2: Trajectory Collection
        agent/trajectory.py → JSONL file
        (complete interaction history for each session)
             ↓
Step 3: Batch Processing
        batch_runner.py → ShareGPT format
        (process multiple prompts in parallel, track tool usage)
             ↓
Step 4: Trajectory Compression
        trajectory_compressor.py
        (protect training signals, compress intermediate history)
             ↓
Step 5: RL Training
        rl_cli.py + environments/ + Atropos
        (generate trainable RL data or SFT data)
             ↓
Step 6: Model Update
        New model replaces old model
             ↓
Step 7: Closed-Loop Verification
        New model performs better on tasks
        → produces better trajectories → better data → ...
```

### Batch Runner

**File**: `batch_runner.py`

```python
# Batch run mode
batch_runner.run(
    prompts_file="prompts.jsonl",
    num_parallel=8,       # parallelism
    model="claude-sonnet-4-6",
    save_trajectories=True,
    tool_stats=True,      # tool usage statistics
    checkpoint_dir="./checkpoints",
)
```

---

## 7. Evolution Mechanism Relationship Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                   Self-Evolution Full Chain Diagram           │
└──────────────────────────────────────────────────────────────┘

  [User Task]
      │
      ▼
  ┌────────────────────────────────────┐
  │           Agent Core Loop           │
  │    run_agent.py / agent/run.py     │
  └──────────────┬─────────────────────┘
                 │
     ┌───────────┼───────────────────────┐
     │           │                       │
     ▼           ▼                       ▼
┌─────────┐ ┌─────────┐           ┌───────────┐
│ Success?│ │  Hit    │           │ Discovered│
│ (>5 calls)│ │  Error? │           │New Workflow?│
└────┬────┘ └────┬────┘           └─────┬─────┘
     │           │                       │
     ▼           ▼                       ▼
  ┌──────────────────────────────────────────┐
  │        skill_manage (action)             │
  │  ┌─────────┐ ┌─────────┐ ┌──────────┐    │
  │  │ CREATE  │ │  PATCH  │ │  CREATE  │    │
  │  │ new skill│ │ patch   │ │ new skill│    │
  │  └────┬────┘ └────┬────┘ └────┬─────┘    │
  └───────┼───────────┼───────────┼──────────┘
          │           │           │
          ▼           ▼           ▼
    ┌────────────────────────────────┐
    │    ~/.hermes/skills/           │
    │    (persistent skill files)    │
    │    + SKILL.md                  │
    │    + references/               │
    │    + templates/                │
    └────────────────────────────────┘
          │
          ▼
    ┌────────────────────────────────┐
    │    prompt_builder.py            │
    │    (skills auto-injected into   │
    │     system prompt)              │
    └────────────────────────────────┘
          │
          ▼
    ┌────────────────────────────────┐
    │    Agent encounters similar task│
    │    → auto-activates relevant    │
    │      skills                     │
    └────────────────────────────────┘

  ┌──────────────────────────────────────────────────┐
  │              Memory Layer (Cross-Session Persistence)│
  │                                                   │
  │  MEMORY.md ── Agent notes ── Environmental facts, project conventions │
  │  USER.md  ── User profile ── Preferences, style, expectations         │
  │  Honcho   ── Semantic memory ── Dialectic understanding, long-term context │
  │  SQLite   ── Session history ── FTS5 full-text search + LLM summary  │
  └──────────────────────────────────────────────────┘
          │
          ▼
    ┌────────────────────────────────┐
    │     session_search tool        │
    │      (cross-session experience reuse) │
    └────────────────────────────────┘

  ┌──────────────────────────────────────────────────┐
  │              RL Training Pipeline (Model-Level Evolution)│
  │                                                   │
  │  trajectory.py ── Trajectory collection          │
  │       ↓                                          │
  │  batch_runner.py ── Batch processing + statistics │
  │       ↓                                          │
  │  trajectory_compressor.py ── Training data compression │
  │       ↓                                          │
  │  rl_cli.py + Atropos ── RL training              │
  │       ↓                                          │
  │  New model ───→ Better Agent ──→ Better trajectories → ... │
  └──────────────────────────────────────────────────┘
```

---

## 8. Technical Implementation Details

### 8.1 Skill Activation Mechanism

Skills are injected into the system prompt via `SKILLS_GUIDANCE` in `prompt_builder.py`:

```python
SKILLS_GUIDANCE = """
After completing a complex task (5+ tool calls), fixing a tricky error,
or discovering a non-trivial workflow, save the approach as a
skill with skill_manage so you can reuse it next time.

When using a skill and finding it outdated, incomplete, or wrong,
patch it immediately with skill_manage(action='patch') - don't wait to be asked.
Skills that aren't maintained become liabilities.
"""
```

This means skill activation is implemented through **LLM reasoning ability** rather than hard-coded rules — the agent autonomously decides whether to create/use/patch skills based on the current task context.

### 8.2 Memory Injection Strategy

Memory is injected via the "frozen snapshot" mode in `memory_tool.py`:

```
┌────────────────────────────────────────────────┐
│  System Prompt (static, loaded once)            │
│  - Contains MEMORY.md snapshot (timestamped)   │
│  - Contains USER.md snapshot                   │
│  - Skill list                                  │
│  - Tool registry                               │
└────────────────────────────────────────────────┘
                      ↓
┌────────────────────────────────────────────────┐
│  Tool Response (dynamic, per call)             │
│  - memory tool returns latest memory content   │
│  - session_search returns latest search results│
│  - honcho_profile returns latest user profile  │
└────────────────────────────────────────────────┘
```

### 8.3 Security Mechanisms

| Security Measure | Implementation Location | Description |
|----------|----------|------|
| Injection attack scanning | `memory_tool.py` | Scan for injection patterns in `MEMORY.md` / `USER.md` |
| Skill security scanning | `tools/skills_hub.py` | Pre-install validation of skill content |
| Skill isolation | `~/.hermes/skills/` | Skills in separate directories, avoiding mutual interference |
| Skill isolation (Hub) | quarantine mechanism | Untrusted skills enter quarantine zone |

### 8.4 Platform Adaptation Layer

**File**: `gateway/platforms/`

```
gateway/
├── run.py              # Unified gateway entry point
└── platforms/
    ├── telegram.py     # Telegram adapter
    ├── discord.py       # Discord adapter
    ├── slack.py         # Slack adapter
    ├── whatsapp.py      # WhatsApp adapter
    └── signal.py        # Signal adapter
```

Each messaging platform's sessions are uniformly stored in SQLite, and session retrieval can span platforms.


---

## 9. Summary & Evaluation

### 9.1 Core Design Philosophy

Hermes Agent's self-evolution mechanism follows the **"Safe Evolution" principle** — instead of modifying the agent's own code, it achieves capability accumulation and reuse through a three-layer persistent knowledge system:

```
Experience → Skills (persistent knowledge) → Cross-session reuse → Better performance
  ↓
Trajectories → Training data → Model update → Stronger capabilities
```

### 9.2 Evolution Mechanism Classification

| Evolution Layer | Specific Mechanism | Reversibility | Scope of Effect |
|----------|----------|--------|----------|
| **Skills Layer** | Create/patch skills | High (editable/deletable) | Specific task types |
| **Memory Layer** | MEMORY.md / USER.md | High (editable) | Agent behavior style |
| **Session Retrieval Layer** | FTS5 search + LLM summary | Medium (overwrites old sessions) | Context understanding |
| **User Modeling Layer** | Honcho profiles | Medium (continuously updated) | Personalized interaction |
| **Trajectory/RL Layer** | Trajectory compression + RL training | Low (model update) | General capabilities |

### 9.3 Innovations

1. **Skills as First-Class Citizens**: Skills are not simple prompt fragments, but complete knowledge modules with versions, platform adaptation, file structure, and reference sub-documents
2. **Frozen Snapshot Mode**: System prompt is stable while memory tool responses are dynamic, solving the contradiction between consistency and freshness
3. **FTS5 + LLM Dual-Layer Retrieval**: Full-text index guarantees speed, LLM summary guarantees relevance
4. **Dialectic User Modeling**: Honcho is not a key-value store, but a system that "dialectically" understands user intent
5. **Trajectory Compression Protects Training Signals**: Not simple truncation, but semantic compression that protects key training signals

### 9.4 Feedback Signal System Full-Scope Analysis

> [!IMPORTANT]
> **This is the core framework for understanding Hermes Agent's self-evolution mechanism: feedback signals, not code structure, are what determine the direction and speed of evolution.**

**The "Explicit-Implicit" Spectrum of Feedback Signals:**

```
Explicit signals (precisely measurable)          Implicit signals (depends on LLM judgment)
        │                                               │
        ▼                                               ▼
┌─────────────────┐                          ┌──────────────────┐
│ 5+ tool call    │                          │ Task "value"     │
│  threshold       │                          │ judgment         │
│ Token count      │                          │ "Non-trivial"    │
│  limit exceeded  │                          │  workflow        │
│ File mtime       │                          │ "Tricky" error   │
│  changes         │                          │  determination   │
│ Tool call        │                          │ Summary          │
│  exceptions      │                          │ "importance"     │
│ Reward function  │                          │  ranking         │
│  return value    │                          │ Honcho profile   │
│                  │                          │  update          │
└────────┬────────┘                          └────────┬─────────┘
         │                                               │
         ▼                                               ▼
   High reproducibility                            High adaptability
   Low latency feedback                             Delayed feedback
   No LLM involvement                              Depends on LLM metacognition
```

**Five Feedback Signal Types and Their Distribution in the System:**

| Signal Type | Typical Examples | Feedback Latency | Manifestation in Hermes |
|---------|---------|---------|----------------|
| **Structural** | Tool call count, token count | Extremely low (millisecond-level) | Skill creation threshold, compression trigger |
| **Outcome** | Task success/failure, tool errors | Low (second-level) | RL reward computation, tool error tracking |
| **Social** | User corrections, user satisfaction | Medium (minute to day) | Honcho conclude, USER.md update |
| **Semantic** | LLM self-assessment, pattern recognition | Medium (depends on LLM speed) | Skill creation decision, summary quality |
| **Environmental** | mtime changes, file existence | Low (filesystem-level) | Skill cache invalidation |

**The Metacognitive Dilemma of Implicit Signals: Who Verifies the Verifier?**
- Skill creation relies on the agent's judgment "is this worth remembering" — but the agent's judgment itself is limited by its cognitive ability
- If the agent **incorrectly underestimates** the value of an experience, it won't create a skill → evolution stagnates
- If the agent **incorrectly overestimates** the value of an experience, it creates a noisy skill → skill repository degrades
- This is a **self-referential feedback loop**: signal quality depends on the system itself, and the system itself is affected by the signal

**The Most Elegant Design: Tool Call Pair Completeness (`_sanitize_tool_pairs`)**
- This is a kind of **implicit constraint feedback**: instead of directly saying "this action is wrong", it says "this action's return value is missing"
- The essence is using **structural completeness** rather than **content correctness** to detect problems
- Comparison with RL: this is equivalent to "valid action space constraint" — the agent can make any mistake, but cannot break the MDP state transition contract

### 9.5 Limitations

1. **No Code-Level Self-Modification**: Evolution is limited to prompts, memory, and skills; does not modify underlying logic
2. **Skill Quality Depends on Agent Judgment**: If the agent makes wrong decisions about "when to create skills", it may produce noisy skills
3. **RL Training Pipeline Requires External Atropos**: Only provides data, does not include complete RL training implementation
4. **No Explicit Evolution Assessment Mechanism**: No quantitative metrics to measure the degree or speed of "evolution"
5. **Skills Not Shared Across Profiles**: Different profiles' skills are completely isolated, cannot inherit

---

## Reference File Index

| File | Purpose |
|------|----------|
| `tools/skill_manager_tool.py` | Skill CRUD core implementation |
| `tools/memory_tool.py` | File memory system |
| `tools/session_search_tool.py` | Cross-session FTS5 retrieval |
| `tools/honcho_tools.py` | Honcho user modeling interface |
| `agent/prompt_builder.py` | System prompt construction & skill injection |
| `agent/context_compressor.py` | Context window compression |
| `agent/trajectory.py` | Trajectory collection |
| `trajectory_compressor.py` | Training data compression |
| `batch_runner.py` | Batch trajectory collection |
| `rl_cli.py` | RL training CLI |
| `environments/hermes_base_env.py` | Atropos RL environment |
| `hermes_state.py` | SQLite session storage |
| `hermes_cli/profiles.py` | Multi-profile isolation |
| `toolsets.py` | Toolset definitions |
| `tools/registry.py` | Tool registry |
