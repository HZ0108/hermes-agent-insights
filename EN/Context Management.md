## 1. Overall Architecture Overview

Hermes Agent's context management system adopts a **layered decoupled** design philosophy, splitting context management into multiple independent modules with clearly defined responsibilities:

```
┌─────────────────────────────────────────────────────────────┐
│                    run_agent.py (AIAgent)                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  System Prompt Builder Layer (prompt_builder.py)    │    │
│  │  ├─ Agent Identity (SOUL.md / DEFAULT_IDENTITY)     │    │
│  │  ├─ Platform Hints (whatsapp/telegram/discord/etc) │    │
│  │  ├─ Memory/Skills/SessionSearch guidance text       │    │
│  │  ├─ Context files (SOUL.md/AGENTS.md/.cursorrules)  │    │
│  │  └─ Skills Index (with LRU + disk snapshot cache)  │    │
│  └─────────────────────────────────────────────────────┘    │
│                              │                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Context References Layer (context_references.py)   │    │
│  │  @file / @folder / @diff / @git / @url             │    │
│  └─────────────────────────────────────────────────────┘    │
│                              │                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Context Compression Layer (context_compressor.py)  │    │
│  │  ├─ Tool output trimming (no LLM call)             │    │
│  │  ├─ Head/Tail protection strategy                   │    │
│  │  ├─ LLM structured summarization                    │    │
│  │  └─ Iterative compression (preserving prior summary)│    │
│  └─────────────────────────────────────────────────────┘    │
│                              │                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Prompt Caching Layer (prompt_caching.py)           │    │
│  │  system_and_3 strategy: 4 cache_control breakpoints  │    │
│  └─────────────────────────────────────────────────────┘    │
│                              │                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Message Format Adaptation Layer (anthropic_adapter)│    │
│  │  OpenAI ↔ Anthropic Messages API bidirectional conv│    │
│  └─────────────────────────────────────────────────────┘    │
│                              │                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Auxiliary Routing Layer (auxiliary_client.py)     │    │
│  │  Unified provider parsing for compression/LLM calls │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. System Prompt Builder Layer (prompt_builder.py)

### 2.1 Layered Structure

The system prompt is composed of multiple independent fragments in the following order:

| Priority | Component | Source | Stability |
|----------|-----------|--------|-----------|
| 1 | Agent Identity | SOUL.md or `DEFAULT_AGENT_IDENTITY` | Static (unchanged within session) |
| 2 | Platform Hints | `PLATFORM_HINTS` dict (whatsapp/telegram/etc) | Static |
| 3 | Memory/Skills Guidance | `MEMORY_GUIDANCE` / `SKILLS_GUIDANCE` etc | Static |
| 4 | Project Context Files | SOUL.md / AGENTS.md / .cursorrules | Static (scanned on demand) |
| 5 | Skills Index | `~/.hermes/skills/` directory scan | Static (cached) |

### 2.2 Context File Priority System

`build_context_files_prompt()` implements a **mutually exclusive priority** mechanism — at any given moment, only one type of project context file is loaded:

```python
project_context = (
    _load_hermes_md(cwd_path)      # Priority 1: .hermes.md / HERMES.md
    or _load_agents_md(cwd_path)   # Priority 2: AGENTS.md
    or _load_claude_md(cwd_path)   # Priority 3: CLAUDE.md
    or _load_cursorrules(cwd_path) # Priority 4: .cursorrules
)
```

### 2.3 Safe Injection: Prompt Threat Scanning

Before context files are injected into the system prompt, they are checked by `_scan_context_content()`:

```python
_CONTEXT_THREAT_PATTERNS = [
    (r'ignore\s+(previous|all|above|prior)\s+instructions', "prompt_injection"),
    (r'do\s+not\s+tell\s+the\s+user', "deception_hide"),
    ...
]
```

Along with zero-width character detection (`\u200b` and other invisible Unicode characters), which are common techniques used in prompt injection attacks.

> [!NOTE]
> **Defensive Design:**
> **Trust but verify**: Hermes adopts a "load then scan" strategy for context files rather than outright rejection. This balances flexibility (allowing users to define custom project rules) with security (blocking obvious injections). Detected threats are logged and replaced with placeholders `[BLOCKED: ...]` rather than crashing outright.
>
> This contrasts with MCP's security philosophy: MCP defaults to rejection, while Hermes defaults to acceptance with verification. In practice, this difference depends on the user scenario — developer tools need flexibility, while production environments may require stricter isolation.

### 2.4 Two-Level Cache for Skills Index

`build_skills_system_prompt()` implements a **two-level cache combining in-process LRU + disk snapshots**:

```
Cache Layer         │ Persistence  │ Invalidation Mechanism
────────────────────┼───────────────┼─────────────────────────────
In-process LRU      │ None          │ Skills file mtime/size change
Disk snapshot       │ Cross-process │ Manifest (mtime+size) mismatch
Filesystem scan     │ Every read    │ None (re-parsed each time)
```

The cache key includes `(skills_dir, external_dirs, available_tools, available_toolsets)`, ensuring cache invalidation when toolsets change.

> [!NOTE]
> **The Art of Cache Invalidation Granularity**: Many systems have cache invalidation logic that is either too coarse-grained (simple TTL) or too complex (dependency injection). Hermes uses a manifest approach with `mtime + size`, which automatically invalidates when files are modified while avoiding the need to parse file contents every time. This is a pragmatic engineering choice.
>
> It is also worth noting that `external_dirs` (external skills directories) have no snapshot cache and must be scanned every time. This means the larger and more numerous the files in external directories, the more linear the cold-start performance degradation.

---

## 3. Context Reference System (context_references.py)

### 3.1 @-Reference Syntax Design

Hermes supports embedding `@` references in messages, with the following formats:

```
@file:path/to/file          # File contents
@file:path/to/file:10-20    # Specific line range in file
@folder:path/to/dir         # Directory structure
@diff                       # git diff
@staged                     # git diff --staged
@git:N                      # git log -N -p
@url:https://...            # Web page contents
```

### 3.2 Security Boundary Control

`preprocess_context_references()` implements a **dual sandbox**:

1. **Path boundary**: The `allowed_root` parameter restricts all relative path resolution to within the specified directory, preventing `../` escape attempts.
2. **Sensitive path interception**: `_ensure_reference_path_allowed()` explicitly blocks:
   - `~/.ssh/*` (private keys, authorized_keys)
   - `~/.hermes/.env` (credentials)
   - 30+ other sensitive file patterns

```python
blocked_exact = {home / rel for rel in _SENSITIVE_HOME_FILES}
blocked_dirs = [home / rel for rel in _SENSITIVE_HOME_DIRS]
```

### 3.3 Token Budget Control

Reference content injection is subject to strict token limits:

- **Hard limit**: Injected content must not exceed **50%** of the context window (direct rejection)
- **Soft limit**: Exceeding **25%** triggers a warning but does not block

```
injected_tokens > 50% context_length → BLOCK (reject injection)
injected_tokens > 25% context_length → WARN (log warning)
```

> [!NOTE]
> **Token budget as the "last mile" safety valve**: Even if all path security checks pass, token budget control ensures that large file references cannot "flood" the conversation context. This is a "Defense in Depth" approach — each layer of checks could theoretically be bypassed, but the combination of multiple layers makes bypassing nearly impossible.
>
> The 25%/50% thresholds are empirical values: In practice, when reference content exceeds a quarter of the context window, conversation quality has already begun to degrade, because the proportion of content the model needs to "recall" increases.

---

## 4. Context Compression System (context_compressor.py)

### 4.1 Compression Triggering Mechanism

The compression system uses a **proactive pre-check + passive threshold** dual trigger:

```python
# Proactive pre-check: rough estimation before API call
def should_compress_preflight(self, messages):
    rough_estimate = estimate_messages_tokens_rough(messages)
    return rough_estimate >= self.threshold_tokens

# Passive threshold: exact token count after API response
def should_compress(self, prompt_tokens=None):
    tokens = prompt_tokens or self.last_prompt_tokens
    return tokens >= self.threshold_tokens
```

The trigger threshold defaults to **50%** of the context window (configurable).

### 4.2 Compression Algorithm: Four-Stage Pipeline

```
Raw message list
    │
    ▼ [Stage 1: Tool Output Trimming]
    No LLM call — directly replace old tool results >200 chars with placeholders
    │
    ▼ [Stage 2: Head/Tail Boundary Determination]
    Head: first N messages (default 3) fully preserved
    Tail: dynamic protection window based on token budget (default ~20K tokens)
    │
    ▼ [Stage 3: LLM Structured Summarization]
    Generate structured summary via LLM for middle turns
    Summaries preserved across iterations (prior results retained)
    │
    ▼ [Stage 4: Tool Pair Integrity Repair]
    Clean up orphan tool_call/tool_result pairs
    (compression may break tool_use → tool_result correspondence)
    │
    ▼ Output
```

> [!NOTE]
> Personal observation: I believe the current processing pipeline has a logical flaw — tool output truncation (Stage 1) should not block knowledge extraction (Stage 3). Context trimming and memory compression should follow parallel processing paths: trimming only applies to the current Prompt assembly, while summarization and memory modules must operate on lossless raw context snapshots. Once tool return results are forcibly truncated on the frontend, the summarization chain receives semantically sparse degraded data, which easily leads to failure in extracting implicit knowledge.

### 4.3 Structured Summary Template

Summaries use a fixed 6-field structure:

```markdown
## Goal
[The user's goal]

## Constraints & Preferences
[User preferences, coding style, constraints, important decisions]

## Progress
### Done  [Completed work]
### In Progress  [In progress]
### Blocked  [Blockers]

## Key Decisions
[Important technical decisions and rationale]

## Relevant Files
[Files read/modified/created]

## Next Steps
[Next steps plan]

## Critical Context
[Critical values, error messages, configurations, etc.]
```

PS: The following viewpoints are not guaranteed to be completely correct.
> [!NOTE]
> **1. Information Loss Caused by Compression and Truncation**
> Compression and truncation are essentially information loss (permanent). To address this, a **Tiered Memory** architecture leveraging the local filesystem with read-write decoupling is suggested: the main routing path aggressively trims to control token budgets, while a parallel asynchronous path persists lossless full interaction traces (including complete tool outputs). Under this mechanism, strong constraint triggers are injected via System Prompt: when structured summary (cache) details are insufficient to support reasoning, the Agent is forced to retrieve from local archives (primary storage) based on indexes, enabling on-demand recall of high-fidelity data and reducing hallucination occurrence.
> ```
> [Memory Access Directive]
> The current context is a lossy compressed structured summary containing only high-level logic and state snapshots. The complete historical trace (including full conversation and uncompressed tool call results) has been persistently archived locally at: [FILE_PATH].
> Action Rule: > When you cannot obtain sufficient details from the current summary to support reasoning, or when you need to verify historical error stacks, long-text parameters, and other implicit knowledge, you MUST NOT hallucinate or blindly infer. You must first invoke the [Search_Tool_Name] tool and search the above archive files using precise keywords to recall the original context before continuing the current task.
> ```
>
> **2. The Problem of "Read" Files**
> If actual content is discarded during compression while the "read" status is preserved, this creates a **dangling reference** that can cause model hallucinations. Methods to eliminate this "state-data inconsistency":
> 1. **On-demand reload**: Preserve status as an index, and when needed, pull actual content back into the context.
> 2. **Cascading erasure**: Ensure WYSIWYG — once content is discarded, the corresponding "read" status is simultaneously deleted.

### 4.4 Tool Pair Integrity Repair

After compression, two types of anomalies may occur:

1. **Orphan tool result**: Tool result exists, but the assistant message containing the call has been compressed → delete the orphaned tool result
2. **Orphan tool call**: Assistant message has `tool_calls`, but the corresponding tool result was truncated → insert placeholder `[Result from earlier conversation — see context summary above]`

```python
# Detect orphan tool results
orphaned_results = result_call_ids - surviving_call_ids
messages = [m for m in messages
            if not (m.get("role") == "tool" and m.get("tool_call_id") in orphaned_results)]

# Add stub tool result
missing_results = surviving_call_ids - result_call_ids
for tc in msg.get("tool_calls"):
    if cid in missing_results:
        patched.append({
            "role": "tool",
            "content": "[Result from earlier conversation — see context summary above]",
            "tool_call_id": cid,
        })
```

### 4.5 Head/Tail Alignment Repair

Compression boundaries may fall in the middle of a tool pair:

- **Tail alignment** (`_align_boundary_backward`): Consecutive tool results before the boundary are included in the compression zone together, avoiding mid-turn truncation
- **Head alignment** (`_align_boundary_forward`): If compression start position is a tool result, slide forward to a non-tool message

> [!NOTE]
> **Message format integrity is a prerequisite for API correctness**: Anthropic's Messages API has strict format requirements for tool calls — each `tool_use` must have a matching `tool_result`, and each `tool_result` must correspond to a `tool_use`. If compression breaks this correspondence, the API returns a 400 error.
>
> A point of confusion: why do incomplete cases occur in the first place? Conventional context management all performs integrity validation — even when compression protection is triggered, it should atomically isolate based on "conversation chunks" to ensure Tool-Call and Result are paired in a closed loop.

---

## 5. Prompt Caching (prompt_caching.py)

### 5.1 The system_and_3 Strategy

Anthropic's Prompt Caching supports `cache_control` markers, which can cache processed content on the server side, **reducing input token costs by 75%**.

```python
# Cache 4 breakpoints: system prompt + last 3 non-system messages
breakpoints_used = 0

if messages[0].get("role") == "system":
    _apply_cache_marker(messages[0], marker)
    breakpoints_used += 1

remaining = 4 - breakpoints_used  # Remaining available breakpoints
non_sys = [i for i in range(len(messages)) if messages[i].get("role") != "system"]
for idx in non_sys[-remaining:]:  # Select from back to front
    _apply_cache_marker(messages[idx], marker)
```

Cache TTL supports `5m` (default) and `1h`.

> [!NOTE]
> PS: TTL uses a **Sliding Expiration** mechanism, calculated from the last active time (Last Active Time). Any new API call triggers a state refresh, resetting and clearing the timer.

### 5.2 Cache Breakpoint Selection Logic

```
[System (cache)] [msg1] [msg2] ... [msgN-3] [msgN-2 (cache)] [msgN-1 (cache)] [msgN (cache)]
                      ↑ Compressed zone (not in cache)
                                        ↑ Last 3 non-system messages enter cache
```

Selection logic: **System message fixed in cache + select remaining breakpoints from the end of the message list**.

> [!NOTE]
> **Balancing dynamic vs. static content**: Anthropic's caching strategy means that the more recent messages are, the less likely they are to be cached (because the cache points are at fixed positions). In Hermes's design, "last 3 non-system messages" may contain content the user just entered, which is反而没有被缓存。
>
> This may be because certain async agents (e.g., CC's memory and compression agents) fork the main agent and deliver task prompts through user blocks — this caching strategy allows both the main agent and other parallel agents to have most content hit the cache.

---

## 6. Message Format Adaptation (anthropic_adapter.py)

### 6.1 OpenAI ↔ Anthropic Bidirectional Conversion

`convert_messages_to_anthropic()` is responsible for converting OpenAI-format messages to Anthropic Messages API format. Core conversions include:

| OpenAI Format | Anthropic Format |
|---------------|------------------|
| `{"role": "system", "content": "..."}` | Extracted as standalone `system` parameter |
| `{"role": "assistant", "content": "...", "tool_calls": [...]}` | Converted to `text` + `tool_use` blocks |
| `{"role": "tool", "tool_call_id": "xxx", "content": "..."}` | Converted to `tool_result` block |
| `{"role": "user", "content": [...]}` (multimodal) | Recursively convert each content block |
| Consecutive same-role messages | Merge (user) or append (assistant) |

### 6.2 OAuth Compatibility Handling

When using a Claude Code OAuth token, `build_anthropic_kwargs()` performs additional compatibility conversions:

1. **System prefix replacement**: `Hermes Agent` → `Claude Code`
2. **Tool name prefix**: `tool_name` → `mcp_tool_name`
3. **Identity masquerading**: Sending Claude Code's User-Agent and version number

```python
if is_oauth:
    block["text"] = block["text"].replace("Hermes Agent", "Claude Code")
    tool["name"] = "mcp_" + tool["name"]
```

> [!NOTE]
> **OAuth adaptation is a textbook case of provider lock-in**: Claude Code's OAuth infrastructure (`platform.claude.com`) requires requests to carry specific `user-agent` and `x-app` headers, otherwise returning a 500 error. This "protocol-level binding" means third-party clients like Hermes must simulate Claude Code's behavior to use OAuth normally.
>
> This reflects a real problem in the API ecosystem: when a provider's (e.g., Anthropic's) OAuth service is tightly coupled with a specific client (e.g., Claude Code), third-party clients must either abandon OAuth (use a regular API key) or precisely simulate client behavior (fragile but functional).

### 6.3 Strict Role Alternation Enforcement

Anthropic's Messages API requires strict role alternation (`user → assistant → user → ...`). `convert_messages_to_anthropic()` implements automatic merging:

```python
# Consecutive assistant messages → merge content blocks
if fixed[-1]["role"] == m["role"] == "assistant":
    fixed[-1]["content"] = prev_blocks + curr_blocks

# Consecutive user messages → merge content text
if fixed[-1]["role"] == m["role"] == "user":
    fixed[-1]["content"] = prev_content + "\n" + curr_content
```

---

## 7. Auxiliary Routing Layer (auxiliary_client.py)

### 7.1 Unified Provider Resolution

`auxiliary_client.py` implements a **single-entry provider resolution chain**. All auxiliary tasks (compression, web extraction, vision analysis, etc.) go through `call_llm()` / `async_call_llm()` calls. The resolution order is:

```
Text task auto-detection chain:
1. OpenRouter (OPENROUTER_API_KEY)
2. Nous Portal (~/.hermes/auth.json)
3. Custom endpoint (config.yaml base_url + OPENAI_API_KEY)
4. Codex OAuth (Responses API → chat.completions adapter)
5. Native Anthropic
6. Direct API-key providers (z.ai/GLM, Kimi, MiniMax)

Vision task auto-detection chain:
1. Primary provider (if it is a supported vision backend)
2. OpenRouter → Nous → Codex → Anthropic → Custom
```

### 7.2 Codex Responses API Adapter

Codex's OAuth token can only access the Responses API (not chat.completions). `_CodexCompletionsAdapter` implements transparent protocol conversion:

```python
# Input: chat.completions format kwargs
{
    "model": "gpt-5.2-codex",
    "messages": [...],
    "tools": [...]
}

# Converted to Responses API format
{
    "model": "gpt-5.2-codex",
    "instructions": "You are a helpful assistant.",
    "input": [{"role": "user", "content": [...]}],
    "tools": [...],
    "store": False,
}

# Output: converted back to chat.completions format
```

### 7.3 Async Event Loop Isolation

Async clients' cache keys include `loop_id`, ensuring each event loop gets an independent client instance:

```python
cache_key = (provider, async_mode, base_url or "", api_key or "", loop_id)
```

This avoids deadlocks caused by reusing `AsyncOpenAI` clients across event loops.

> [!NOTE]
> **Event loop affinity is a hidden trap in async SDKs**: `httpx.AsyncClient` is bound to the event loop it was created in, but in background tasks (`run_in_background`), the event loop may have already switched. Hermes ensures each loop gets an independent client instance by including `loop_id` in the cache key.
>
> This problem does not occur in short-lived scripts, but it becomes a stability killer in gateway mode (long-running with multiple concurrent requests).

---

## 8. Security Layer (redact.py)

### 8.1 Multi-Layer Redaction Strategy

`redact_sensitive_text()` implements 8 types of sensitive information detection patterns, in execution order:

```python
# ── 1. Known-prefix API Keys (23 providers) ────────────────────────────
# Word boundary lookbehind (?<!/!) prevents false matches of substrings in normal text
_PREFIX_PATTERNS = [
    r"sk-[A-Za-z0-9_-]{10,}",           # OpenAI / Anthropic / OpenRouter
    r"ghp_[A-Za-z0-9]{10,}",            # GitHub PAT (classic)
    r"github_pat_[A-Za-z0-9_]{10,}",    # GitHub PAT (fine-grained)
    r"xox[baprs]-[A-Za-z0-9-]{10,}",    # Slack tokens
    r"AIza[A-Za-z0-9_-]{30,}",          # Google API keys
    r"AKIA[A-Z0-9]{16}",                # AWS Access Key ID (fixed length)
    r"sk_live_[A-Za-z0-9]{10,}",        # Stripe live key
    r"sk_test_[A-Za-z0-9]{10,}",        # Stripe test key
    r"SG\.[A-Za-z0-9_-]{10,}",          # SendGrid
    r"hf_[A-Za-z0-9]{10,}",             # HuggingFace
    r"tvly-[A-Za-z0-9]{10,}",           # Tavily
    ...  # 23 total, including Perplexity / Fal.ai / Replicate / npm / PyPI / DigitalOcean etc.
]
_PREFIX_RE = re.compile(
    r"(?<![A-Za-z0-9_-])(" + "|".join(_PREFIX_PATTERNS) + r")(?![A-Za-z0-9_-])"
)

# ── 2. Environment variable assignments KEY=value (with quotes) ─────────
# Matches all variables with names containing API_?KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|AUTH, case-insensitive
_SECRET_ENV_NAMES = r"(?:API_?KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH)"
_ENV_ASSIGN_RE = re.compile(
    rf"([A-Z_]*{_SECRET_ENV_NAMES}[A-Z_]*)\s*=\s*(['\"]?)(\S+)\2",
    re.IGNORECASE,
)

# ── 3. JSON fields "apiKey": "value" ──────────────────────────────────
# Covers 12 common field names: apiKey / token / password / secret / access_token etc.
_JSON_KEY_NAMES = r"(?:api_?[Kk]ey|token|secret|password|access_token|\
refresh_token|auth_token|bearer|secret_value|raw_secret|secret_input|key_material)"
_JSON_FIELD_RE = re.compile(rf'("{_JSON_KEY_NAMES}")\s*:\s*"([^"]+)"', re.IGNORECASE)

# ── 4. Authorization request header ────────────────────────────────────
_AUTH_HEADER_RE = re.compile(r"(Authorization:\s+Bearer\s+)(\S+)", re.IGNORECASE)

# ── 5. Telegram Bot Token (both formats supported) ──────────────────────
# bot123456789:ABCDEF...   or   123456789:ABCDEF...
_TELEGRAM_RE = re.compile(r"(bot)?(\d{8,}):([-A-Za-z0-9_]{30,})")

# ── 6. PEM private key blocks (RSA / EC / OPENSSH and all PKCS#8 variants) ──
_PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN[A-Z ]*PRIVATE KEY-----[\s\S]*?-----END[A-Z ]*PRIVATE KEY-----"
)

# ── 7. Database connection strings (protocol://user:PASSWORD@host) ────
# Supports postgres(postgresql) / mysql / mongodb(+srv) / redis / amqp
_DB_CONNSTR_RE = re.compile(
    r"((?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|amqp)://[^:]+:)([^@]+)(@)",
    re.IGNORECASE,
)

# ── 8. E.164 international phone numbers (Signal / WhatsApp etc.) ──────
# Negative lookahead (?![A-Za-z0-9]) prevents misidentifying hex strings as phone numbers
_SIGNAL_PHONE_RE = re.compile(r"(\+[1-9]\d{6,14})(?![A-Za-z0-9])")
```

### 8.2 Token Retention Strategy

Short tokens (<18 characters): fully masked → `***`
Long tokens: preserve first 6 + last 4 → `sk-abcde...ghij`

```python
def _mask_token(token: str) -> str:
    if len(token) < 18:
        return "***"
    return f"{token[:6]}...{token[-4:]}"
```

> [!NOTE]
> **Redaction completeness determines the security floor**: The core challenge of regex-based redaction is the trade-off between **coverage vs. false positive rate**. Hermes chose relatively conservative patterns (e.g., `sk-[A-Za-z0-9_-]{10,}` requiring at least 10 characters), which means it won't accidentally redact ordinary `sk`-prefixed strings, but may miss non-standard format keys.
>
> The most critical blind spot: `redact.py` is runtime redaction (applied to log output), not compile-time or transport-time redaction. This means if the API request itself is intercepted, redaction protects nothing. The correct use case is: log file leaks, accidental exposure in print output, debug information propagation, and other "post-hoc leak" scenarios.

---

## 9. Token Budget Management (model_metadata.py)

### 9.1 Model Context Length Resolution Chain

`get_model_context_length()` implements a 10-level fallback resolution:

```
0. Explicit config override (user explicitly set)
   ↓ Not satisfied
1. Persistent cache (disk yaml, model@base_url as key)
   ↓ Not satisfied
2. Endpoint /models metadata (custom endpoints of non-well-known providers only)
   ↓ Not satisfied
3. Local server query (Ollama/LM Studio/vLLM/llama.cpp)
   ↓ Not satisfied
4. Anthropic /v1/models API (requires sk-ant-api*, OAuth not supported)
   ↓ Not satisfied
5. Provider-aware models.dev query
   ↓ Not satisfied
6. OpenRouter real-time API metadata
   ↓ Not satisfied
7. Fuzzy hardcoded default values (matched by model family)
   ↓ Not satisfied
8. Query local server (last resort)
   ↓ Not satisfied
9. Default fallback: 128K
```

### 9.2 Local Server Type Detection

`detect_local_server_type()` identifies local inference servers by probing known endpoints:

```
LM Studio → GET /api/v1/models → check response structure
Ollama    → GET /api/tags → check "models" field
llama.cpp → GET /v1/props → check "default_generation_settings"
vLLM     → GET /version → check "version" field
```

> [!NOTE]
> **The cost of incorrect context length is extremely high**: Hermes's 10-level fallback chain finds the correct value in most common scenarios, but for new models not in the list or uncommon providers, it eventually falls back to 128K, which could result in a 1M context model being treated as 128K (premature compression).
>
> An ideal design should provide user feedback mechanisms: when the fallback value significantly differs from the actual context window (e.g., API returns context exceeded error), the user should be prompted to set `model.context_length`.

---

## 10. Comprehensive Design Philosophy Analysis

### 10.1 Modularity and Testability

Each module (`prompt_builder`, `context_compressor`, `context_references`, `anthropic_adapter`, `prompt_caching`, `auxiliary_client`) is **stateless or a pure function**, with dependencies injected via parameters rather than global state. This allows each module to be independently unit-tested, and changes like "switch compression model" or "add new @reference type" only require modifying the corresponding module.

### 10.2 Progressive Degradation Strategy

The entire system is full of "if X fails, try Y; if Y also fails, use Z" degradation chains:

- Provider resolution: OpenRouter → Nous → Custom → Codex → Anthropic → API-key
- Context length: config → cache → endpoint metadata → hardcoded → 128K fallback
- Compression summary: LLM summary → middle messages discarded directly (no summary injected)

The last type of degradation (middle messages discarded directly) is the least ideal, but the system does not crash as a result.

### 10.3 Security Over Performance

When balancing security checks and performance optimization, Hermes chooses the more secure direction:
- Context files are threat-scanned before injection (even if it costs a small amount of CPU)
- @references have strict token limits (preventing context flooding)
- Sensitive paths are explicitly blocked (rather than allowed by default)
- Prompt Caching breakpoint count is constrained by Anthropic limits (rather than attempting to break limits)

> [!NOTE]
> **The fundamental contradiction between security and performance**: In the field of context management, security checks (scanning, budget control, path validation) are naturally enemies of performance — they require extra computation and storage. Scanning context files increases system prompt construction time; token budget checks require pre-estimating content token counts; @reference resolution requires checking file access permissions.
>
> Hermes's strategy is "security operations executed once at build time, not repeated at runtime." This is reasonable in CLI scenarios (each request is an independent system prompt construction), but in ultra-low-latency streaming scenarios, the cost of these security checks needs to be re-evaluated.
