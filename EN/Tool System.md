# Hermes Agent Tool System: In-Depth Analysis Report

## Table of Contents

1. [Overview](#1-overview)
    - [1.1 Scope of Analysis](#11-scope-of-analysis)
    - [1.2 Tool Architecture Overview](#12-tool-architecture-overview)
2. [Core Architecture Overview](#2-core-architecture-overview)
    - [2.1 Import Chain Design](#21-import-chain-design)
    - [2.2 ToolEntry Data Structure](#22-toolentry-data-structure)
3. [Tool Registry (ToolRegistry)](#3-tool-registrytoolregistry)
    - [3.1 Singleton Pattern](#31-singleton-pattern)
    - [3.2 Core Method Analysis](#32-core-method-analysis)
    - [3.3 Query Interfaces](#33-query-interfaces)
4. [Toolsets System](#4-toolsets-system)
    - [4.1 Layered Structure Design](#41-layered-structure-design)
    - [4.2 Recursive Resolution of resolve_toolset()](#42-recursive-resolution-of-resolve_toolset)
    - [4.3 Platform-Specific Toolsets](#43-platform-specific-toolsets)
    - [4.4 Dynamic Registration of Plugin Toolsets](#44-dynamic-registration-of-plugin-toolsets)
5. [Tool Orchestration Layer (model_tools.py)](#5-tool-orchestration-layermodel_toolspy)
    - [5.1 Tool Discovery Mechanism](#51-tool-discovery-mechanism)
    - [5.2 Three-Layer Async Bridging Design](#52-three-layer-async-bridging-design)
    - [5.3 Dynamic Schema Patching](#53-dynamic-schema-patching)
6. [Tool Call Parsers](#6-tool-call-parsers)
    - [6.1 Background](#61-background)
    - [6.2 Parser Registration Mechanism](#62-parser-registration-mechanism)
    - [6.3 Hermes Parser Example](#63-hermes-parser-example)
7. [MCP (Model Context Protocol) Support](#7-mcp-model-context-protocol-support)
    - [7.1 Architectural Positioning](#71-architectural-positioning)
    - [7.2 MCP Client Core Design](#72-mcp-client-core-design)
    - [7.3 Security Design](#73-security-design)
    - [7.4 Sampling Mechanism](#74-sampling-mechanism)
    - [7.5 Reconnection and Fault Tolerance](#75-reconnection-and-fault-tolerance)
8. [Plugin System](#8-plugin-system)
    - [8.1 Three-Source Plugin Discovery](#81-three-source-plugin-discovery)
    - [8.2 PluginContext Facade](#82-plugincontext-facade)
    - [8.3 Lifecycle Hooks](#83-lifecycle-hooks)
    - [8.4 Disable Mechanism](#84-disable-mechanism)
9. [Code Execution Sandbox](#9-code-execution-sandbox)
    - [9.1 Sandbox Toolset Injection](#91-sandbox-toolset-injection)
    - [9.2 Multiple Execution Backends](#92-multiple-execution-backends)
10. [Lifecycle Hooks and Interceptors](#10-lifecycle-hooks-and-interceptors)
    - [10.1 Agent Loop Tool Interception](#101-agent-loop-tool-interception)
    - [10.2 Read/Write Tool Continuity Detection](#102-readwrite-tool-continuity-detection)
11. [Toolset Distribution System](#11-toolset-distribution-system)
    - [11.1 Tool Probability Distributions in RL Training](#111-tool-probability-distributions-in-rl-training)
12. [Trade-offs in Architectural Decisions](#12-trade-offs-in-architectural-decisions)
    - [12.1 Singleton Registry vs Dependency Injection](#121-singleton-registry-vs-dependency-injection)
    - [12.2 Silent Failure vs Strict Checking](#122-silent-failure-vs-strict-checking)
    - [12.3 check_fn Granularity](#123-check_fn-granularity)
    - [12.4 Tool Schema Versioning](#124-tool-schema-versioning)
13. [Security Design Analysis](#13-security-design-analysis)
    - [13.1 Defense in Depth](#131-defense-in-depth)
    - [13.2 Limitations of Credential Scrubbing](#132-limitations-of-credential-scrubbing)

---

## 1. Overview

### 1.1 Scope of Analysis

This report covers the following core subsystems:

| Subsystem | File Path | Core Responsibility |
|-----------|-----------|---------------------|
| Tool Registry | `tools/registry.py` | Tool registration, storage, and distribution |
| Toolsets System | `toolsets.py` | Tool grouping and dynamic composition |
| Tool Orchestration Layer | `model_tools.py` | Tool discovery, Schema generation, and call dispatch |
| Tool Call Parsers | `environments/tool_call_parsers/` | Multi-model format parsing |
| MCP Client | `tools/mcp_tool.py` | External MCP server integration |
| MCP Server | `mcp_serve.py` | Exposing Agent as an MCP service |
| Plugin System | `hermes_cli/plugins.py` | Third-party extension mechanism |
| Code Execution Sandbox | `tools/code_execution_tool.py` | Controlled code execution environment |
| Toolset Distribution | `toolset_distributions.py` | Tool probability distributions in RL training |

### 1.2 Tool Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Agent Entry Layer                        │
│          run_agent.py / cli.py / batch_runner.py                │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Tool Orchestration Layer model_tools.py         │
│  get_tool_definitions()  ←→  handle_function_call()              │
└──────────┬────────────────┬──────────────────────┬──────────────┘
           │                │                      │
           ▼                ▼                      ▼
┌─────────────────┐  ┌─────────────┐  ┌──────────────────────────┐
│   Tool Registry  │  │ Toolset     │  │  MCP Client (mcp_tool)  │
│  registry.py     │  │ toolsets.py │  │  plugins.py (plugins)   │
└────────┬────────┘  └──────┬──────┘  └────────────┬─────────────┘
         │                   │                       │
         ▼                   ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Tool Implementation Layer (tools/)            │
│  file_tools  web_tools  terminal_tool  browser_tool             │
│  vision_tools  code_execution_tool  memory_tool  ...             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Core Architecture Overview

### 2.1 Import Chain Design

Hermes Agent's tool system adopts a **zero-coupling centralized registration** pattern, where tool modules self-register at import time. The circular import-safe design of the entire chain is as follows:

```
tools/registry.py          ← No dependencies; leaf node of the entire system
        ↑
tools/*.py                 ← Depends only on registry; calls register() at import
        ↑
model_tools.py             ← Depends on registry + toolsets
        ↑
run_agent.py / cli.py      ← Depends on model_tools; entry node
```

### 2.2 ToolEntry Data Structure

```python
class ToolEntry:
    __slots__ = (
        "name",        # Unique tool name
        "toolset",     # Belonging toolset
        "schema",      # OpenAI format Schema
        "handler",     # Handler function
        "check_fn",    # Availability check function
        "requires_env",# Required environment variables
        "is_async",    # Whether it's an async tool
        "description", # Tool description
        "emoji",       # UI display emoji
    )
```


---

## 3. Tool Registry (ToolRegistry)

### 3.1 Singleton Pattern

```python
registry = ToolRegistry()  # Module-level singleton
```

Using a module-level singleton instead of a Dependency Injection (DI) container represents a balance between language conventions and pragmatism. In CLI tool scenarios, the complexity of DI containers far outweighs the benefits; however, this plants a hidden danger of singleton state for testing and MCP remote invocation scenarios.

### 3.2 Core Method Analysis

#### register() — Tool Registration

```python
def register(
    self, name, toolset, schema, handler,
    check_fn=None, requires_env=None, is_async=False,
    description="", emoji=""
):
```

> [!NOTE]
> **Design Insight:**
>
> - **Idempotent override**: Duplicate registrations with the same name emit a WARNING rather than throwing an exception, allowing MCP to dynamically re-register tools. This is critical when MCP server tool lists change (`notifications/tools/list_changed`).
> - **toolset check_fn merging**: Each toolset stores only the first registered tool's check_fn; subsequent check_fns for the same toolset are silently ignored. This means all tools within a toolset must share the same availability check logic — designers must ensure tool and toolset check_fn semantics are aligned.
> - **`is_async` flag**: Tools self-declare async capability, and the registry transparently bridges the sync/async boundary at dispatch time. This flag avoids the overhead of runtime introspection.

#### get_definitions() — Schema Aggregation and Filtering

```python
def get_definitions(self, tool_names: Set[str], quiet: bool = False) -> List[dict]:
```

This is the interface between the tool system and LLM APIs. Key behaviors:

1. **check_fn caching**: The `check_results` dictionary caches check_fn results within a single invocation, preventing the same check function from being executed multiple times.
2. **Always fills in the name field**: Even if the Schema lacks a `name`, it falls back to `entry.name`, preventing the LLM from receiving nameless tool definitions.
3. **Returns OpenAI format**: `{"type": "function", "function": {...}}` format directly interfaces with the OpenAI API.

#### dispatch() — Tool Execution

```python
def dispatch(self, name: str, args: dict, **kwargs) -> str:
    entry = self._tools.get(name)
    if not entry:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        if entry.is_async:
            from model_tools import _run_async
            return _run_async(entry.handler(args, **kwargs))
        return entry.handler(args, **kwargs)
    except Exception as e:
        return json.dumps({"error": f"Tool execution failed: {type(e).__name__}: {e}"})
```

> [!NOTE]
> **Design Insight:**
>
> - **Exception encapsulation**: All exceptions are caught and formatted into `{"error": "..."}` JSON strings. This design fixes the exception handling boundary at the registry layer, ensuring the Agent never crashes due to tool exceptions.
> - **Unified string return value**: Regardless of what type a tool returns, it's ultimately serialized to a string, simplifying the Agent Loop's message construction. However, there is a potential issue here — if a handler returns a dictionary, after `json.dumps()` it is passed to the LLM, which sees a double-serialized string that may affect parsing.
> - **Transparent async bridging**: `_run_async` is called from the registry layer rather than the handler layer, ensuring handler authors don't need to care about sync/async concerns.

#### deregister() — Dynamic Tool Unloading

```python
def deregister(self, name: str) -> None:
    entry = self._tools.pop(name, None)
    if entry.toolset in self._toolset_checks and not any(
        e.toolset == entry.toolset for e in self._tools.values()
    ):
        self._toolset_checks.pop(entry.toolset, None)
```

The core mechanism for MCP dynamic tool discovery. When an MCP server's tool list changes, the entire toolset is cleared and re-registered ("nuke-and-repave" strategy). This is a brute-force but effective approach, avoiding the complexity of incremental updates.

### 3.3 Query Interfaces

| Method | Purpose |
|--------|---------|
| `get_all_tool_names()` | List all tools |
| `get_schema(name)` | Get raw Schema (bypassing check_fn) |
| `get_toolset_for_tool(name)` | Query which toolset a tool belongs to |
| `get_emoji(name)` | Get UI emoji |
| `is_toolset_available(toolset)` | Check toolset availability |
| `check_tool_availability()` | Batch check, returns (available list, unavailable details) |

---

## 4. Toolsets System

### 4.1 Layered Structure Design

Toolsets implement a lightweight **tool composition algebra** through a two-layer structure of `tools` + `includes`:

```python
TOOLSETS = {
    "web": {
        "description": "Web research...",
        "tools": ["web_search", "web_extract"],
        "includes": []
    },
    "safe": {
        "description": "Safe toolkit...",
        "tools": ["mixture_of_agents"],
        "includes": ["web", "vision", "image_gen"]  # Composes other toolsets
    },
}
```

**Direct tools (tools)** are leaf nodes, while **inclusion relationships (includes)** allow toolsets to reference each other, forming a **diamond dependency**:

```
safe toolset
├── tools: [mixture_of_agents]
└── includes: [web, vision, image_gen]
                ├── web: [web_search, web_extract]
                └── vision: [vision_analyze]
```

> [!NOTE]
> **Optimistic parsing over strict validation**: `resolve_toolset()` uses a silent-skip strategy for circular dependencies, returning an empty list rather than throwing an exception. The code comments explicitly distinguish between the two: diamond dependencies (`safe → web → file` and `debugging → file` converging) are legal and normal — silent skipping avoids false positives; genuine rare circular references are considered "low-probability events, safe to skip." This design chooses **availability** (not crashing on config errors) over **correctness** (not warning on cycles). This contrasts with some language module systems that refuse to load when cycles are detected — safer but more fragile. In the toolset configuration scenario, Hermes's choice is reasonable because toolset definition errors should be caught by tests and CI, not runtime crashes.

### 4.2 Recursive Resolution of resolve_toolset()

```python
def resolve_toolset(name: str, visited: Set[str] = None) -> List[str]:
    if name in {"all", "*"}:  # Global alias
        all_tools: Set[str] = set()
        for toolset_name in get_toolset_names():
            resolved = resolve_toolset(toolset_name, visited.copy())
            all_tools.update(resolved)
        return list(all_tools)

    if name in visited:  # Cycle detection
        return []  # Silent skip

    visited.add(name)
    toolset = TOOLSETS.get(name)
    tools = set(toolset.get("tools", []))
    for included_name in toolset.get("includes", []):
        tools.update(resolve_toolset(included_name, visited))

    return list(tools)
```

**Design Insight:**

- **"all"/"*" alias**: Uses a special string as the "all tools" alias and uses `visited.copy()` to create an independent visitation history for each branch. This detail is crucial — without copy, the same toolset encountered via a different path would trigger a false cycle detection.
- **Silent cycle skipping**: Returns an empty list on cycles rather than throwing an exception. The comment explains: diamond dependencies are legal (tools have already been collected via another path), and genuine rare cycles can also be safely skipped. This is an **optimistic parsing** strategy that avoids cycle detection breaking normal functionality.
- **`visited.copy()` special handling in "all" alias**: Ensures that during global resolution, the visited set remains independent on each branch, preventing cross-branch contamination.

### 4.3 Platform-Specific Toolsets

The codebase defines 13 messaging platform toolsets (telegram, discord, whatsapp, slack, etc.), which **share the same `_HERMES_CORE_TOOLS` list**:

```python
_HERMES_CORE_TOOLS = [
    "web_search", "web_extract",
    "terminal", "process",
    "read_file", "write_file", "patch", "search_files",
    "vision_analyze", "image_generate",
    "mixture_of_agents",
    "skills_list", "skill_view", "skill_manage",
    "browser_navigate", "browser_snapshot", ...,
    "todo", "memory",
    "session_search", "clarify",
    "execute_code", "delegate_task",
    "cronjob",
    "send_message",
    "honcho_context", "honcho_profile", ...,
    "ha_list_entities", "ha_get_state", ...,
]
```

Security policy differences between messaging platforms (such as dangerous command approval mechanisms for terminal tools) are implemented at the handler layer via `check_fn` and `dangerous command approval`, rather than at the toolset layer. This keeps toolsets' pure tool-grouping semantics intact.

### 4.4 Dynamic Registration of Plugin Toolsets

```python
def _get_plugin_toolset_names() -> Set[str]:
    return {
        entry.toolset
        for entry in registry._tools.values()
        if entry.toolset not in TOOLSETS
    }
```

This mechanism allows plugins to **inject new toolset names** into the `TOOLSETS` dictionary at runtime. Both `get_all_toolsets()` and `validate_toolset()` are aware of these dynamically added entries, making plugin-provided toolsets completely equal to built-in toolsets at the interface level.

> [!NOTE]
> **What "runtime registration" actually means**: The "registration" here refers to capabilities within the Python process, **not an LLM-discoverable dynamic discovery mechanism**. What truly notifies the LLM of tool changes at runtime is the **MCP's `notifications/tools/list_changed` event**: Hermes's MCP client subscribes to this event, triggering `deregister` + re-registration, so the latest Schema is sent down on the next LLM API call. Hermes's local plugin registration does not have this mechanism — toolset changes require a process restart to take effect.
> This design may consider KV cache and policy consistency issues: suddenly telling the model "you have X more tools" creates a cognitive conflict: should it continue the current plan (not using X), or re-plan (using X)? This could cause it to make new decisions that contradict already-executed steps, especially when the agent loop is in the middle of multi-step operations.

---

## 5. Tool Orchestration Layer (model_tools.py)

### 5.1 Tool Discovery Mechanism

```python
def _discover_tools():
    _modules = [
        "tools.web_tools",
        "tools.terminal_tool",
        "tools.file_tools",
        ...
    ]
    for mod_name in _modules:
        try:
            importlib.import_module(mod_name)
        except Exception as e:
            logger.warning("Could not import tool module %s: %s", mod_name, e)
```

**Design Insight:**

- **Silent failure strategy**: Optional tools (such as `image_generation_tool` failing to import due to missing `fal_client`) don't crash the entire system. This is very useful when developing experimental tools, but can also mask real dependency issues.
- **Hardcoded module list**: The tool module list is maintained in `model_tools.py`; adding new tool modules requires modifying this list. Although toolsets.py can dynamically sense registered tools, the discovery phase still requires explicit declaration. This limitation is acceptable in practice.

### 5.2 Three-Layer Async Bridging Design

The `_run_async()` function handles the complex scenario of synchronously calling async tools:

```python
def _run_async(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Scenario 1: In existing async context → Start new thread
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result(timeout=300)

    if threading.current_thread() is not threading.main_thread():
        # Scenario 2: In worker thread → Use thread-local persistent loop
        worker_loop = _get_worker_loop()
        return worker_loop.run_until_complete(coro)

    # Scenario 3: In main thread sync context → Use global persistent loop
    tool_loop = _get_tool_loop()
    return tool_loop.run_until_complete(coro)
```

**Three-scenario handling logic:**

| Scenario | Trigger Condition | Solution | Why |
|----------|-------------------|----------|-----|
| In async context | `loop.is_running()` is True | `ThreadPoolExecutor` new thread | Don't block current event loop |
| In worker thread | Non-main thread | Thread-local persistent loop | Avoid GC issues from `asyncio.run()`'s create/destroy lifecycle |
| In main thread sync | Default path | Global persistent loop | Reuse loop to avoid "Event loop is closed" |

### 5.3 Dynamic Schema Patching

```python
# Dynamic schema patching for execute_code
if "execute_code" in available_tool_names:
    sandbox_enabled = SANDBOX_ALLOWED_TOOLS & available_tool_names
    dynamic_schema = build_execute_code_schema(sandbox_enabled)
    # Replace original schema

# Description cleanup for browser_navigate
if "web_search" not in available_tool_names:
    # Remove "prefer web_search" wording from browser_navigate description
```

> [!NOTE]
> **Design Intent**: Prevent tool name hallucination — when an LLM sees a tool name mentioned in a Schema, it tends to invoke it even if that tool is actually unavailable.

---

## 6. Tool Call Parsers

### 6.1 Background

Different model families have vastly different output formats for tool calls:

| Model | Format |
|-------|--------|
| Hermes Custom Fine-tuned | `<tool_call>{"name": "...", "arguments": {...}}</tool_call>` |
| Llama | Various JSON structures |
| Qwen | Special XML tags |
| Mistral | Dedicated format |
| DeepSeek | V3 / V3.1 different formats |
| GLM | GLM-4.5 / GLM-4.7 different formats |
| Kimi | K2 format |

### 6.2 Parser Registration Mechanism

```python
class ToolCallParser(ABC):
    @abstractmethod
    def parse(self, text: str) -> ParseResult:
        raise NotImplementedError

PARSER_REGISTRY: Dict[str, Type[ToolCallParser]] = {}

def register_parser(name: str):
    def decorator(cls):
        PARSER_REGISTRY[name] = cls
        return cls
    return decorator

@register_parser("hermes")
class HermesToolCallParser(ToolCallParser):
    ...
```

### 6.3 Hermes Parser Example

```python
PATTERN = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>|<tool_call>\s*(.*)", re.DOTALL
)

def parse(self, text: str) -> ParseResult:
    if "<tool_call>" not in text:
        return text, None

    matches = self.PATTERN.findall(text)
    tool_calls = []
    for match in matches:
        raw_json = match[0] if match[0] else match[1]
        if not raw_json.strip():
            continue
        tc_data = json.loads(raw_json)
        tool_calls.append(ChatCompletionMessageToolCall(
            id=f"call_{uuid.uuid4().hex[:8]}",
            type="function",
            function=Function(
                name=tc_data["name"],
                arguments=json.dumps(tc_data.get("arguments", {})),
            ),
        ))

    content = text[:text.find("<tool_call>")].strip()
    return content if content else None, tool_calls
```


---

## 7. MCP (Model Context Protocol) Support

### 7.1 Architectural Positioning

MCP support uses a **bidirectional mirror** architecture:

```
Hermes Agent
    ├── tools/mcp_tool.py     → MCP Client: connects to external MCP servers (GitHub, Filesystem, etc.)
    └── mcp_serve.py          → MCP Server: exposes Hermes tools to Claude Code, Cursor, and other clients
```

This allows Hermes to be both a **consumer** of MCP tools and a **provider** of MCP tools.

### 7.2 MCP Client Core Design

#### Transport Layer Support

```python
# Stdio transport: subprocess method
from mcp.client.stdio import stdio_client
async with stdio_client(config) as (read, write):
    async with ClientSession(stdio_interface) as session:
        ...

# HTTP transport: remote server
from mcp.client.streamable_http import streamablehttp_client
async with streamablehttp_client(url, headers=...) as (read, write, get_error):
    ...
```

#### Command Resolution Enhancement

```python
def _resolve_stdio_command(command: str, env: dict) -> tuple[str, dict]:
    """Resolves bare commands (npx/npm/node) to absolute paths"""
    if os.sep not in resolved_command:
        which_hit = shutil.which(resolved_command, path=path_arg)
        if which_hit:
            resolved_command = which_hit
```

### 7.3 Security Design

#### Environment Variable Filtering

```python
_SAFE_ENV_KEYS = frozenset({
    "PATH", "HOME", "USER", "LANG", "LC_ALL", "TERM", "SHELL", "TMPDIR",
})

def _build_safe_env(user_env: Optional[dict]) -> dict:
    env = {}
    for key, value in os.environ.items():
        if key in _SAFE_ENV_KEYS or key.startswith("XDG_"):
            env[key] = value
    if user_env:
        env.update(user_env)
    return env
```

Only whitelisted environment variables plus user-explicitly-configured variables are passed, preventing API keys/tokens from leaking into MCP subprocesses.

#### Credential Scrubbing

```python
_CREDENTIAL_PATTERN = re.compile(
    r"ghp_[A-Za-z0-9_]{1,255}"
    r"|sk-[A-Za-z0-9_]{1,255}"
    r"|Bearer\s+\S+"
    r"|token=[^\s&,;\"']{1,255}"
    ...
)
```

Credential patterns in error messages are replaced with `[REDACTED]`, preventing sensitive information from being exposed by MCP server error responses.

### 7.4 Sampling Mechanism

MCP servers can proactively request LLM sampling (`sampling/createMessage`). The SamplingHandler implementation:

```python
class SamplingHandler:
    def __init__(self, server_name: str, config: dict):
        self.max_rpm = config.get("max_rpm", 10)  # Requests per minute
        self.max_tool_rounds = config.get("max_tool_rounds", 5)  # Tool loop round limit
        self.model_override = config.get("model")  # Model override
        self.allowed_models = config.get("allowed_models", [])  # Whitelist

    def _check_rate_limit(self) -> bool:
        # Sliding window rate limiter
        now = time.time()
        window = now - 60
        self._rate_timestamps[:] = [t for t in self._rate_timestamps if t > window]
        ...
```

**Multi-layer protection:**

- **Rate limiting**: Prevents MCP servers from exhausting LLM quotas
- **Model whitelist**: Only pre-approved models process Sampling requests
- **Tool loop upper limit**: Prevents Sampling requests from triggering infinite tool call loops
- **Per-instance state**: All state in SamplingHandler is instance-level (`_rate_timestamps`, `metrics`), with no module-level global state, supporting multiple MCP servers running independently

### 7.5 Reconnection and Fault Tolerance

```python
_MAX_RECONNECT_RETRIES = 5
_MAX_BACKOFF_SECONDS = 60
```

Uses **exponential backoff reconnection** with a maximum wait of 60 seconds. The error message parsing function `_format_connect_error()` recursively traverses nested exception chains, extracting the innermost `FileNotFoundError` and providing actionable error hints (e.g., "missing executable 'npx'").

> [!NOTE]
> **Thread Safety of MCP Async Daemon Thread**: The MCP client (`mcp_tool.py`) runs an independent event loop in a daemon thread. All tool calls are safely scheduled onto this loop from external threads via `run_coroutine_threadsafe()`. This solves the runtime model conflict between the tool system (synchronous entry) and the MCP session (async API) — there's no need to asynchronize the entire tool layer, only to isolate this one async subsystem. This is a classic **thread-isolated decoupling** pattern: the most complex async part is encapsulated in a single thread, presenting a synchronous interface externally. The downside is the overhead of a resident thread and the recovery complexity when the daemon thread crashes.

---

## 8. Plugin System

### 8.1 Three-Source Plugin Discovery

```python
# 1. User plugins: ~/.hermes/plugins/<name>/
# 2. Project plugins: ./.hermes/plugins/<name>/
# 3. Pip plugins: packages with hermes_agent.plugins entry-point
```

### 8.2 PluginContext Facade

```python
class PluginContext:
    def register_tool(self, name, toolset, schema, handler, ...):
        from tools.registry import registry
        registry.register(...)

    def register_hook(self, hook_name: str, callback):
        self._manager._hooks.setdefault(hook_name, []).append(callback)

    def inject_message(self, content, role):
        """Injects a message into the current conversation context"""
        self._manager._injected_messages.append({"role": role, "content": content})
```

> [!NOTE]
> Plugins should not know the specific implementation details of the underlying registry. PluginContext provides a **Facade Pattern**, encapsulating multiple subsystems (registry, hooks, message injection) behind a single interface.

### 8.3 Lifecycle Hooks

```python
VALID_HOOKS = {
    "pre_tool_call",     # Before tool execution
    "post_tool_call",    # After tool execution
    "pre_llm_call",      # Before LLM call
    "post_llm_call",     # After LLM call
    "on_session_start",  # At session start
    "on_session_end",    # At session end
}
```

The hook mechanism provides the core capability of a **middleware layer**, enabling plugins to:

- Validate parameters before tool execution (e.g., security scanning)
- Log audit logs after tool execution
- Inject system prompts before LLM calls
- Modify responses after LLM calls

### 8.4 Disable Mechanism

```python
def _get_disabled_plugins() -> set:
    config = load_config()
    disabled = config.get("plugins", {}).get("disabled", [])
    return set(disabled)
```

Through the `plugins.disabled` field in `config.yaml`, users can disable problematic plugins without downtime.

> [!NOTE]
> **Engineering Trade-offs of Progressive Skill Loading**: The Skills system uses a Progressive Disclosure strategy: `skills_list` returns only metadata (name ≤64 characters, description ≤1024 characters), and `skill_view` loads the full instruction content. This contrasts with MCP's approach of exposing all tool schemas at once — MCP's approach is more LLM-friendly (no extra tool call needed) but has higher token costs; Hermes's approach is token-efficient but adds interaction rounds. YAML frontmatter simultaneously serves as manifest and security policy carrier (`platforms` restricts loading platforms, `prerequisites` declares runtime dependencies), requiring no additional configuration layer. This co-location design of metadata and content reduces the drift risk between skill descriptions and actual code.

---

## 9. Code Execution Sandbox

### 9.1 Sandbox Toolset Injection

```python
if "execute_code" in available_tool_names:
    sandbox_enabled = SANDBOX_ALLOWED_TOOLS & available_tool_names
    dynamic_schema = build_execute_code_schema(sandbox_enabled)
```

The `execute_code` tool allows the LLM to generate and execute Python code. This generated code can invoke other Hermes tools, thereby **significantly reducing LLM interaction rounds** (multiple tools can be called in sequence within a single code block).

Key design: The available tools in the code execution environment are determined by **externally passed-in toolsets**, not the global registry. This allows the Agent Loop to precisely control sub-agent tool permissions.

### 9.2 Multiple Execution Backends

Code execution supports multiple isolation environments:

```
tools/environments/
    ├── base.py              # Abstract base class
    ├── docker.py            # Docker container isolation
    ├── local.py             # Direct execution (dev mode only)
    ├── daytona.py           # Daytona commercial sandbox
    ├── modal.py             # Modal cloud functions
    ├── ssh.py               # Remote SSH execution
    ├── singularity.py       # Singularity containers
    └── persistent_shell.py  # Persistent shell sessions
```

This multi-backend design reflects Hermes Agent's **production-grade ambition**: different deployment scenarios require different isolation levels, from local direct execution during development to Docker/Modal isolation in production.

> [!NOTE]
> **Sandbox Tool Injection: Explicit Parameterization vs Global Lookup**: The available toolset for the execute_code tool is explicitly injected via the `enabled_tools` parameter at the call site, rather than querying the global registry inside the handler. This design choice is critical: it prevents sub-agents from indirectly calling unauthorized tools from the parent session by executing arbitrary code. If using a global query, a restricted sub-agent started by `delegate_task` might inherit the parent session's full toolset, causing a permission escape. Fixing permission boundaries on the call stack is a lightweight form of **capability space isolation** — no process-level isolation needed, just correct parameter passing.

---

## 10. Lifecycle Hooks and Interceptors

### 10.1 Agent Loop Tool Interception

```python
_AGENT_LOOP_TOOLS = {"todo", "memory", "session_search", "delegate_task"}

def handle_function_call(...):
    if function_name in _AGENT_LOOP_TOOLS:
        return json.dumps({"error": f"{function_name} must be handled by the agent loop"})
```

These tools are registered with the registry but return errors during dispatch, forcing the Agent Loop to handle them directly. Reason: **These tools need access to Agent Loop-level state** (TodoStore, MemoryStore, etc.), which is lost if dispatched through the registry.

This is a **framework-enforced rather than convention-constrained** design: preventing direct calls via error messages rather than code paths. In theory, direct calls to the registry are still possible, but the LLM receives a clear error message.

### 10.2 Read/Write Tool Continuity Detection

```python
_READ_SEARCH_TOOLS = {"read_file", "search_files"}

def handle_function_call(...):
    if function_name not in _READ_SEARCH_TOOLS:
        notify_other_tool_call(task_id)
```

Continuously reading files (above a threshold) triggers a warning, preventing the Agent from endlessly reading files in a loop without making modifications. This is an **implicit loop detection** — rather than directly detecting loops, it detects the imbalance between "intentional behavior" (modifying files) and "lazy behavior" (continuous reading).

---

## 11. Toolset Distribution System

### 11.1 Tool Probability Distributions in RL Training

```python
DISTRIBUTIONS = {
    "default": {
        "description": "All available tools, all the time",
        "toolsets": {
            "web": 100, "vision": 100, "image_gen": 100,
            "terminal": 100, "file": 100, "moa": 100, "browser": 100
        }
    },
    "image_gen": {
        "description": "Heavy focus on image generation...",
        "toolsets": {
            "image_gen": 90, "vision": 90, "web": 55,
            "terminal": 45, "moa": 10
        }
    },
}
```

During reinforcement learning training, different toolset distributions allow the Agent to learn tool usage strategies for different scenarios. The "image_gen" distribution reduces the probability of "moa" (reasoning tool), allocating more training budget to image generation-related scenarios.

> [!NOTE]
> **Hidden Mismatch Between RL Training Distributions and Production Environments**: The toolset probability distributions used during training directly shape the Agent's behavioral strategy — it learns to preferentially select certain tools when they're available. But in production deployment, if the actually enabled toolsets differ from the training distributions (e.g., terminal has a weight of 100 in the "default" distribution but is disabled at deployment), the Agent may behave inconsistently with training. This is a **deployment environment shift (distribution shift)** problem that is more insidious in RL-based Agents than in supervised learning: the model doesn't just generate different outputs — it calls different tool combinations under different tool availability conditions. The introduction of toolset_distributions shows the designers recognized this — but ensuring training distributions and production configurations remain in sync is still a manually maintained engineering constraint.


---

## 12. Trade-offs in Architectural Decisions

### 12.1 Singleton Registry vs Dependency Injection

**Choice**: Module-level singleton `registry = ToolRegistry()`

**Trade-offs**:

| Pros | Cons |
|------|------|
| No DI container needed; simple and direct | Testing requires mocking or patching global state |
| No performance overhead for global access | In MCP multi-process scenarios, registry state is not shared |
| Great developer experience with import-and-use | |

In CLI tool (single-process, synchronous) scenarios, singleton is a completely reasonable choice. But for MCP multi-server architecture, each `MCPServerTask` runs on an independent event loop sharing the same registry instance — while this currently works, multi-process scaling in the future will require redesign.

### 12.2 Silent Failure vs Strict Checking

When tool module imports fail, only a WARNING is issued rather than an ERROR:

```python
try:
    importlib.import_module(mod_name)
except Exception as e:
    logger.warning("Could not import tool module %s: %s", mod_name, e)
```

**Trade-offs**:

- **Pros**: Missing optional dependencies (such as fal_client for image generation) don't prevent the system from starting; experimental tools can be introduced without affecting the main system.
- **Cons**: The root cause of missing tools (not installed, import error, version conflict) is hidden; `Warning` level logs may be completely ignored under default configuration, leaving users unaware that a tool is unavailable.

### 12.3 check_fn Granularity

Currently `check_fn` is at the **toolset level** availability check, not the tool level:

```python
if check_fn and toolset not in self._toolset_checks:
    self._toolset_checks[toolset] = check_fn
```

This means all tools in the same toolset share one check_fn. If a toolset contains 5 tools but only 2 require a certain API key, the current design lets the 2 tools' check_fn control the availability of all 5 tools. In practice, Hermes's toolset division is granular enough (typically 1-3 tools per toolset) that this limitation has minimal impact, but it constrains further toolset composition.

### 12.4 Tool Schema Versioning

Currently there is no Schema versioning mechanism — when tool parameters change, the Schema updates with the code, and the LLM gets the latest Schema via API calls. This is viable in continuous deployment environments but may cause issues in model fine-tuning scenarios: old-version fine-tuned models may generate parameters incompatible with the new Schema.

---

## 13. Security Design Analysis

### 13.1 Defense in Depth

Hermes Agent's tool system implements multi-layered security controls:

| Layer | Mechanism | Protection Target |
|-------|-----------|-------------------|
| Transport security | MCP environment variable filtering | Credential leakage into subprocesses |
| Content security | Credential scrubbing regex | Credential exposure in error messages |
| Platform security | `check_fn` | Platform-specific tool availability |
| Interaction security | Dangerous command approval | Dangerous commands on messaging platforms |
| Sandbox security | Multi-backend isolation | Code execution isolation |
| Plugin security | Disabled list | Disabling problematic plugins |
| Schema security | Dynamic cross-reference cleanup | Tool name hallucination |

### 13.2 Limitations of Credential Scrubbing

While regex scrubbing covers common credential formats, blind spots remain:

- Credentials in environment variables (such as `$GITHUB_TOKEN`) are not scrubbed
- Custom API keys in non-standard formats
- Structured credentials in MCP server responses (tokens in JSON body)

This is a **known imperfect solution**: perfect credential scrubbing requires semantic understanding, while regex can only cover pattern matching. The designers chose "cover as many common patterns as possible" rather than "attempt perfection but risk over-scrubbing legitimate content."

> [!NOTE]
> **Tirith: Dual-Layer Architecture of Binary and Regex Scanning**: Hermes has two complementary security scanning mechanisms: `skills_guard.py` uses pure Python regex for static analysis of skill files, targeting known threat patterns (data exfiltration, prompt injection, etc.); `tirith_security.py` calls an external Rust binary tool for deeper semantic scanning. Both share the same strategy: **fail-open** — when tirith times out or is unavailable, it only logs a WARNING and does not block operations. This chooses a pragmatic path between security and availability: better to let suspicious content through than to let the security tool itself become a single point of failure. The auto-install mechanism (downloading via SHA-256 checksum with cosign supply-chain verification) ensures tirith is always available, but also means the security boundary depends on the tirith version released by Hermes not being tampered with — an external binary supply chain risk.

---

## Reference File Index

| File | Lines | Description |
|------|-------|-------------|
| `tools/registry.py` | 276 | Core registry |
| `toolsets.py` | 642 | Toolset definitions and resolution |
| `model_tools.py` | 473 | Tool orchestration layer |
| `tools/mcp_tool.py` | ~600+ | MCP client |
| `mcp_serve.py` | — | MCP server |
| `hermes_cli/plugins.py` | ~200+ | Plugin system |
| `environments/tool_call_parsers/` | — | 11 Parser implementations |
| `tools/code_execution_tool.py` | — | Code execution sandbox |
| `toolset_distributions.py` | — | RL tool distribution |
| `environments/agent_loop.py` | — | Agent main loop |
