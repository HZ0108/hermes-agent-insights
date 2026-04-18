# Hermes Agent — Intent Routing

## Table of Contents

1. [1. Overall Architecture Overview](#1-overall-architecture-overview)
2. [2. Command Routing — The Core Intent Routing Layer](#2-command-routing--the-core-intent-routing-layer)
   - [2.1 Central Registry](#21-central-registry)
   - [2.2 Command Parsing and Dispatch](#22-command-parsing-and-dispatch)
   - [2.3 ACP Adapter Command Routing](#23-acp-adapter-command-routing)
   - [2.4 Skill Command Routing](#24-skill-command-routing)
   - [2.5 Quick Commands](#25-quick-commands-user-defined-fast-commands)
3. [3. Model Routing](#3-model-routing--smart-cheap-vs-strong)
   - [3.1 Overview](#31-overview)
   - [3.2 Keyword Heuristics](#32-keyword-heuristics-simple-intent-detection)
   - [3.3 Routing Decision Logic](#33-routing-decision-logic)
   - [3.4 Routing Integration Entry Point](#34-routing-integration-entry-point)
4. [4. Provider Routing](#4-provider-routing)
   - [4.1 OpenRouter Provider Preference Routing](#41-openrouter-provider-preference-routing)
   - [4.2 Runtime Provider Resolution](#42-runtime-provider-resolution)
5. [5. Message Delivery Routing](#5-message-delivery-routing)
   - [5.1 DeliveryRouter](#51-deliveryrouter)
   - [5.2 Delivery Target Resolution Rules](#52-delivery-target-resolution-rules)
6. [6. Sub-Agent Delegation Routing](#6-sub-agent-delegation-routing)
7. [7. Platform Adapter Routing](#7-platform-adapter-routing)
   - [7.1 Architecture](#71-architecture)
   - [7.2 Platform Adapter Registration](#72-platform-adapter-registration)
8. [8. Hook System](#8-hook-system--extensible-event-routing)
9. [9. CLI-Side Routing](#9-cli-side-routing)
10. [10. Summary](#10-summary-hermes-intent-routing-overview)
    - [Key Design Principles](#key-design-principles)

---

## 1. Overall Architecture Overview

Intent Routing in Hermes Agent is **not** a standalone module or an ML-driven intent recognition system. Instead, it is a **multi-layered, multiplexed, hierarchical decision-making mechanism**. After a user message enters the system, routing decisions are made at different levels:

| Layer | Routing Target | Driven By | File Location |
|-------|----------------|-----------|---------------|
| Command Routing | Built-in commands starting with `/` vs. Skill commands vs. Quick Commands | String prefix matching + command registry | `gateway/run.py`, `acp_adapter/server.py` |
| Model Routing | Choose strong model (default) or cheap model | Keyword heuristic rules | `agent/smart_model_routing.py` |
| Message Routing | Which Messaging Platform Adapter to deliver to | Session Source tracking | `gateway/delivery.py` |
| Tool Routing | Which tool the LLM decides to call | LLM's own decision | `tools/registry.py` |
| Sub-Agent Routing | Which sub-agent/provider to delegate tasks to | Configuration-driven | `tools/delegate_tool.py` |
| Platform Routing | Which messaging platform an event originates from | Event Source parsing | `gateway/run.py` |

---

## 2. Command Routing — The Core Intent Routing Layer

### 2.1 Central Registry

**File**: `hermes_cli/commands.py`

This is the **Single Source of Truth** for the entire command routing system.

```python
# Data structure
@dataclass(frozen=True)
class CommandDef:
    name: str                          # Canonical name, no slash: "background"
    description: str                   # Human-readable description
    category: str                      # Category: "Session", "Configuration", etc.
    aliases: tuple[str, ...] = ()       # Aliases: ("bg",)
    args_hint: str = ""                # Argument hint: "<prompt>", "[name]"
    subcommands: tuple[str, ...] = ()  # Tab-completable subcommands
    cli_only: bool = False             # Available in CLI only
    gateway_only: bool = False         # Available in Gateway/messaging platforms only
    gateway_config_gate: str | None = None  # Configuration gate
```

All commands are registered in the `COMMAND_REGISTRY` list as `CommandDef` entries. Key built-in command categories:

**Session**:
- `/new`, `/reset` — New session
- `/retry` — Retry last message
- `/undo` — Remove last exchange
- `/compress` — Manually compress conversation context
- `/background` (`/bg`) — Run prompt in background asynchronously
- `/btw` — Ephemeral side question (no tools, not persisted)
- `/queue` (`/q`) — Queue a message (does not interrupt current run)
- `/rollback` — Restore filesystem checkpoint
- `/stop` — Terminate all background processes
- `/approve`, `/deny` — Approve/deny dangerous commands

**Configuration**:
- `/reasoning` — Manage reasoning effort and display
- `/verbose` — Toggle tool progress display mode
- `/yolo` — Skip all dangerous command approval
- `/personality` — Set predefined personality
- `/provider` — Show current provider

**Tools & Skills**:
- `/skills` — Search/install/view skills
- `/cron` — Manage scheduled tasks
- `/reload-mcp` — Reload MCP servers
- `/browser` — Connect browser tool
- `/plugins` — List installed plugins

**Info**:
- `/commands` — Browse all commands paginated
- `/help` — Show help
- `/usage` — Display token usage
- `/insights` — Usage insight analytics
- `/platforms` — Gateway platform status

### 2.2 Command Parsing and Dispatch (Gateway Layer)

**File**: `gateway/run.py` — `_handle_message()` method (~line 1678)

This is the entry method where the Gateway receives events from various messaging platforms (telegram, discord, slack, etc.). The message processing pipeline is as follows:

```
Receive MessageEvent
    │
    ▼
1. User authorization check (_is_user_authorized)
    │
    ▼
2. Running Agent conflict check
    ├─ If agent is running → Special handling for /stop, /new, /queue
    │   ├─ /stop → Hard termination
    │   ├─ /new → Interrupt + reset session
    │   └─ /queue → Append to queue, no interrupt
    └─ No running agent → Continue
    │
    ▼
3. Extract command: event.get_command()
    │
    ▼
4. Command alias resolution: resolve_command() → CommandDef.name
    │
    ▼
5. if-branch matching on canonical command names
    ├─ Built-in commands → Corresponding _handle_*_command() method
    ├─ Skill commands → build_skill_invocation_message()
    ├─ Quick Commands (user-defined) → Execute directly
    └─ Plugin commands → get_plugin_command_handler()
    │
    ▼
6. No command matches → Send message as normal user input to LLM
```

#### 2.2.1 `MessageEvent.get_command()` — Command Extraction

**File**: `gateway/platforms/base.py` — `MessageEvent` class

```python
class MessageEvent:
    text: str
    message_type: MessageType = MessageType.TEXT
    source: SessionSource = None

    def is_command(self) -> bool:
        return self.text.startswith("/")

    def get_command(self) -> Optional[str]:
        """Extract command name from messages starting with `/`."""
        if not self.is_command():
            return None
        parts = self.text.split(maxsplit=1)
        raw = parts[0][1:].lower() if parts else None
        if raw and "@" in raw:
            raw = raw.split("@", 1)[0]  # Handle Telegram @mention format
        return raw

    def get_command_args(self) -> str:
        """Get arguments after the command."""
        if not self.is_command():
            return self.text
        parts = self.text.split(maxsplit=1)
        return parts[1] if len(parts) > 1 else ""
```

**Routing decision point**: This is a pure string operation that checks `text.startswith("/")` to determine if a message is a command. No ML or complex NLP required.

> [!NOTE]
> **Command routing is essentially a Deterministic Finite Automaton (DFA)**
> **Negative Routing Pattern**: Hermes's command dispatch has a design principle opposite to most intent classification systems: **only "unrecognized" commands flow to the LLM**. This is negative routing (or "LLM as final fallback").

#### 2.2.2 Command Alias Resolution

**File**: `hermes_cli/commands.py` — `resolve_command()` function

```python
def resolve_command(name: str) -> CommandDef | None:
    """Resolve a command name or alias to a CommandDef."""
    return _COMMAND_LOOKUP.get(name.lower().lstrip("/"))

# Build lookup table
_COMMAND_LOOKUP: dict[str, CommandDef] = _build_command_lookup()
# Includes command names and all aliases (e.g., "bg" → CommandDef("background", ...))
```

**Key**: In `gateway/run.py`, aliases are resolved to canonical names before dispatch:
```python
_cmd_def = _resolve_cmd(command) if command else None
canonical = _cmd_def.name if _cmd_def else command
# All subsequent if-checks use the canonical name
```


#### 2.2.3 `GATEWAY_KNOWN_COMMANDS` — Gateway-Recognizable Command Set

```python
GATEWAY_KNOWN_COMMANDS: frozenset[str] = frozenset(
    name
    for cmd in COMMAND_REGISTRY
    if not cmd.cli_only or cmd.gateway_config_gate  # Include all gateway-side commands
    for name in (cmd.name, *cmd.aliases)
)
```

Use cases:
1. **Telegram BotCommands menu** — Telegram only supports `/command` format menus
2. **Slack subcommand mapping** — `slack_subcommand_map()` generates slash handler mappings
3. **Command completion** — `SlashCommandCompleter` provides tab completion

### 2.3 ACP Adapter Command Routing (Headless Scenario)

**File**: `acp_adapter/server.py`

The ACP (Agent Client Protocol) adapter handles messages from editors (e.g., Claude Code) in headless scenarios:

```python
class HermesACPAgent(acp.Agent):
    # Intercept slash commands (no LLM call needed)
    if user_text.startswith("/"):
        response_text = self._handle_slash_command(user_text, state)
        if response_text is not None:
            # Return text response directly, skip LLM
            return PromptResponse(stop_reason="end_turn")

    # Unrecognized command → send to LLM (user may have typed /something as text input)
```

Built-in headless command list:
```python
_SLASH_COMMANDS = {
    "help": "Show available commands",
    "model": "Show or change current model",
    "tools": "List available tools",
    "context": "Show conversation context info",
    "reset": "Clear conversation history",
    "compact": "Compress conversation context",
    "version": "Show Hermes version",
}
```

> [!NOTE]
> The headless scenario is a server environment with "no GUI, no audio device, no interactive terminal." Hermes Agent gracefully degrades in this scenario through environment detection, lazy loading, safe wrappers, etc., ensuring it continues to run properly rather than crashing.

### 2.4 Skill Command Routing (Dynamic Commands)

**File**: `agent/skill_commands.py` — `scan_skill_commands()`

Skills are stored as `SKILL.md` files in `~/.hermes/skills/`, and each skill is automatically registered as a slash command:

```python
def scan_skill_commands() -> Dict[str, Dict[str, Any]]:
    """Scan skills/ directory and return a mapping of /command-name → skill info."""
    # Walk ~/.hermes/skills/**/*.md
    # Parse name and description from frontmatter
    # Command name = name.lower().replace(' ', '-')
    # Register to _skill_commands global dict
```

**Skill command dispatch flow** (`gateway/run.py` lines 2028–2048):

Skills are stored as `SKILL.md` files in `~/.hermes/skills/`. During system startup, `scan_skill_commands()` scans and registers them. The command name is derived from the `name` field in frontmatter (spaces replaced with `-`, lowercased).

```python
if command:
    from agent.skill_commands import get_skill_commands, build_skill_invocation_message
    skill_cmds = get_skill_commands()   # Returns {"skill-name": {...}, ...}
    cmd_key = f"/{command}"
    if cmd_key in skill_cmds:           # Skill command hit
        user_instruction = event.get_command_args().strip()
        msg = build_skill_invocation_message(cmd_key, user_instruction, task_id=_quick_key)
        if msg:
            event.text = msg           # Replace message content with skill prompt
        # Continue fall through → hand to LLM (skill content as prompt)
    else:
        # Attempt to match "known but disabled/not installed" skill → give actionable hint
        _unavail_msg = _check_unavailable_skill(command)
        if _unavail_msg:
            return _unavail_msg
```

Below is the complete Skill command dispatch flowchart:

```
User sends /<skill-name> [args...]
         │
         ▼
┌────────────────────────────────────────────┐
│  gateway/run.py:_handle_message()          │
│  Extract command = "skill-name"            │
└─────────────────┬──────────────────────────┘
                  │
                  ▼
    command is non-empty → Check Skill command table
         │
         ▼
┌────────────────────────────────────────────┐
│  get_skill_commands()                       │
│  (lazy scan ~/.hermes/skills/, cached)     │
└─────────────────┬──────────────────────────┘
                  │
                  ▼
        cmd_key = f"/{command}"
                  │
         ┌────────┴────────┐
         │ cmd_key in      │
         │ skill_cmds ?     │
         └────────┬────────┘
            Yes   │    No
     ┌───────────┴────────────┐
     ▼                        ▼
┌──────────────┐    ┌─────────────────────────────┐
│ Skill match   │    │ _check_unavailable_skill() │
│              │    │ Check for disabled/not-      │
│              │    │ installed skill with same   │
│              │    │ name                         │
│              │    └─────────────┬───────────────┘
│              │                  │
│              │           ┌──────┴──────────┐
│              │           │ Unavailable    │
│              │           │ skill exists?   │
│              │           └──────┬──────────┘
│              │              Yes │    No
│              │           ┌──────┴──────┐
│              │           ▼             ▼
│              │     ┌──────────┐   (fall through
│              │     │Return    │    → hand to LLM)
│              │     │hint msg,  │
│              │     │guide user│
│              │     │to install │
│              │     └──────────┘
│              ▼
│  build_skill_invocation_message(cmd_key, user_args)
│              │
│              ▼
│  ┌───────────────────────────────────────────┐
│  │ _load_skill_payload(skill_dir, task_id)    │
│  │   ↓                                       │
│  │ Call tools.skills_tool.skill_view()        │
│  │   ↓                                       │
│  │ Read SKILL.md content + linked_files      │
│  │ Return (loaded_skill, skill_dir, skill_name) │
│  └─────────────────┬─────────────────────────┘
│                    │
│                    ▼
│  _build_skill_message(loaded_skill, ...)
│  Concatenate formatted skill message:         │
│    [SYSTEM: The user has invoked the "X"     │
│     skill, indicating they want you to follow│
│     its instructions. ...]                    │
│    <SKILL.md body>                           │
│    [setup notes if needed]                  │
│    [supporting files list]                   │
│    The user has provided the following       │
│    instruction: <args>                       │
│                    │                        │
│                    ▼                        │
│         event.text = msg                    │
│         (replace original message content)  │
│                    │                        │
└────────────────────┼─────────────────────────┘
                     │
                     ▼
          (Fall through, continue normal message processing)
                     │
                     ▼
    ┌─────────────────────────────────────────┐
    │ AIAgent.run_conversation(event)          │
    │ Skill content as system prompt / first   │
    │ message. LLM executes task per skill     │
    │ instructions.                            │
    └─────────────────────────────────────────┘
```

### 2.5 Quick Commands (User-Defined Fast Commands)

**File**: `gateway/run.py` lines 1971–2010

Users can define fast commands in `config.yaml` that require no LLM call:

```yaml
quick_commands:
  hello:
    type: exec
    command: "echo Hello from Hermes!"
  greet:
    type: alias
    target: /background
```

- `type: exec` → Execute shell command directly, return output
- `type: alias` → Alias to another command, continue dispatch flow

---

## 3. Model Routing — Smart Cheap vs. Strong

### 3.1 Overview

Based on **surface-level features** of user input (length, keywords, etc.), it decides at the **per-turn conversation level** whether to use the **cheap model** or the **default strong model**.

**File**: `agent/smart_model_routing.py`

> [!NOTE]
> **Per-Turn Independent Decision vs. Cross-Turn State — Motivation for Abandoning MDP Modeling**
>
> `choose_cheap_model_route()` makes decisions **independently for each message** ("stateless per-turn"), without introducing cross-turn conversation hidden state (e.g., "last turn used cheap model and user didn't complain → continue using cheap"). This is a deliberate choice to forgo a stronger model (Markov Decision Process / Bandit).
>
> From an algorithmic perspective, cross-turn state modeling could capture users' "implicit feedback" (e.g., whether the user immediately follows up or complains) to dynamically adjust routing, but the costs are: (1) significantly increased implementation complexity; (2) need for additional instrumentation and feedback collection mechanisms; (3) state space explosion requiring online learning or rule engine maintenance.
>
> The current independent decision model is equivalent to assuming each turn's message complexity is independent from other turns — an oversimplification, certainly, but in practice, as long as most users' simple questions are indeed resolved within a single turn, the error introduced by this simplification is acceptable.

### 3.2 Keyword Heuristics (Simple Intent Detection)

```python
_COMPLEX_KEYWORDS = {
    "debug", "debugging", "implement", "refactor", "patch",
    "traceback", "stacktrace", "exception", "error",
    "analyze", "analysis", "investigate",
    "architecture", "design", "compare", "benchmark", "optimize",
    "review", "terminal", "shell", "tool", "tools",
    "pytest", "test", "tests",
    "plan", "planning", "delegate", "subagent",
    "cron", "docker", "kubernetes",
}
```

Any keyword match triggers the strong model; no match combined with satisfying length constraints leads to the cheap model. This set is manually maintained and fully interpretable, but coverage is limited by maintainer experience (e.g., includes "debug" but not "troubleshoot").

### 3.3 Routing Decision Logic

```python
def choose_cheap_model_route(user_message: str, routing_config) -> Optional[Dict]:
    """If the message looks simple, return the configured cheap model; otherwise return None (use primary model)."""

    # Decision conditions (ALL must be met to use cheap model):
    if len(text) > max_chars (default 160): return None
    if word_count > max_words (default 28): return None
    if newline_count > 1: return None
    if "```" in text or "`" in text: return None  # Code
    if URL_RE.search(text): return None           # URL
    if words & _COMPLEX_KEYWORDS: return None      # Complex keywords

    # None of the above triggered → use cheap model
    return cheap_model_config
```

**Characteristics**:
- **Conservative strategy**: Better to mistakenly use the strong model than to fail on a complex task with the cheap model
- **Zero-shot**: Entirely rule-based, no training whatsoever
- **Configurable**: `cheap_model` and threshold parameters configured in `config.yaml`

> [!NOTE]
> **`words & _COMPLEX_KEYWORDS` — Set Intersection as a Multi-Feature Classifier**
>
> The algorithm is equivalent to a 0-1 weighted linear classifier (hit = 1, threshold = 1):
> - **Advantage**: Fully interpretable — just check two sets to understand the routing result, no logging needed
> - **Drawback**: Morphologically fragile — "analyze" is in the set, "analysis" is not; morphological variants cause missed detections

> [!NOTE]
> **The Essence of the Conservative Strategy is an Extreme Choice in the Precision-Recall Tradeoff**
>
> The costs of the two types of errors are severely asymmetric:
> - **Miss (complex → cheap)**: Task failure, incorrect tool calls, user must retry → High cost
> - **False alarm (simple → strong model)**: Extra cost → Low cost
>
> Therefore, thresholds are set very tightly (`max_chars=160`, `max_words=28`, any code blocks rejected outright), equivalent to the optimal conservative boundary in Bayesian decision theory with asymmetric losses.

> [!NOTE]
> **Survivorship Bias Trap in the Keyword Set**
>
> Words in the set = words "observed to appear in complex tasks"; words not in the set may:
> 1. Truly be strongly correlated with simple tasks
> 2. Also appear in complex tasks but were not sampled
> 3. Be ambiguous (e.g., "test": unit test requests are complex, casual conversation is simple)
>
> **Suggestion**: Run periodic offline analysis, compute P(simple|keyword), and use Precision/Recall curves to evaluate word additions/removals.

### 3.4 Routing Integration Entry Point

**File**: `gateway/run.py` lines 779–791

```python
def _resolve_turn_agent_config(self, user_message: str, model: str, runtime_kwargs: dict) -> dict:
    from agent.smart_model_routing import resolve_turn_route
    primary = {"model": model, ...}
    return resolve_turn_route(user_message, self._smart_model_routing, primary)
```

Called once per message turn, making an independent routing decision each time.

---

## 4. Provider Routing

### 4.1 OpenRouter Provider Preference Routing

**File**: `gateway/run.py` lines 996–1007, `hermes_cli/runtime_provider.py`

Loaded from `config.yaml`:

```yaml
provider_routing:
  # Specify preferred OpenRouter provider for specific models
  # e.g., force certain models through a designated provider for better pricing/availability
```

### 4.2 Runtime Provider Resolution

**File**: `hermes_cli/runtime_provider.py` — `resolve_runtime_provider()`

Supports unified resolution across multiple providers:
- `openrouter` — OpenRouter aggregation platform
- `nous` — Nous Research
- `zai` — Zai platform
- `kimi-coding` — Kimi Coding
- `minimax` — MiniMax
- `openai` — OpenAI Direct
- `custom` — User-defined provider

---

## 5. Message Delivery Routing

### 5.1 DeliveryRouter

**File**: `gateway/delivery.py` — `DeliveryRouter` class

Resolves delivery targets for cron task outputs and agent responses:

```python
@dataclass
class DeliveryTarget:
    platform: Platform        # telegram, discord, slack, feishu, ...
    chat_id: Optional[str] = None      # None → use home channel
    thread_id: Optional[str] = None
    is_origin: bool = False             # Whether to reply to origin
    is_explicit: bool = False           # Whether explicitly specified

    @classmethod
    def parse(cls, target: str, origin: Optional[SessionSource]) -> "DeliveryTarget":
        """Parse delivery target string."""
        # "origin" → reply to origin
        # "local" → local file
        # "telegram" → Telegram home channel
        # "telegram:123456789" → specified chat
```

> [!NOTE]
> **Three-Key Deduplication — Trade-off Between Business Invariance and Implementation Simplicity**
>
> Deduplication key = `(platform, chat_id, thread_id)` tuple; `is_origin` is NOT included in the key:
> - `["telegram", "origin"]` where origin = telegram → deduplicated, send once ✓
> - `["telegram", "origin"]` where origin = discord → both retained ✓
>
> The string `"origin"` is normalized to `origin.platform` in `to_string()`, equivalent to an explicit platform — this is **semantic normalization**: different strings representing the same target are unified before delivery, avoiding duplicate sends due to different origin markers.

> [!NOTE]
> **Hop-by-Hop Append vs. Global Deduplication — Write Order Dependency Risk**
>
> `always_log_local=True` + `"local"` already in delivery targets → local is appended twice (once in loop, once at end). The implementation prevents duplicate delivery via the deduplication key, but the two-append logic itself is confusing.
>
> **Suggestion**: Use `seen_platforms` to handle `always_log_local` uniformly in the loop, eliminating the implicit end-of-loop append behavior.

> [!NOTE]
> **Platform Message Truncation is Defensive Programming, Not a Routing Decision**
>
> `MAX_PLATFORM_OUTPUT = 4000` is proactive trimming to comply with downstream platform API limits:
> - **Positioning**: `DeliveryRouter`'s responsibility is "where to deliver", not "how large it can deliver"
> - **Strategy**: Different platforms have different rate limits; precise limits are known only at runtime; defensive trimming is the most robust degradation
> - **Guarantee**: Truncation point includes a pointer (`full output saved to {path}`); message content is not lost
>
> Consistent with the "conservative strategy" (rather not use cheap model):宁可丢失尾部，也不愿消息整体投递失败.

### 5.2 Delivery Target Resolution Rules

```python
# Supported target formats:
# "origin"       → Reply to the originating messaging platform
# "local"        → Save to local file only
# "telegram"     → Telegram home channel
# "telegram:123456" → Specified Telegram chat
# ["local", "telegram"] → Multiple targets
```

Cron jobs use this mechanism in their `deliver` field.

---

## 6. Sub-Agent Delegation Routing

**File**: `tools/delegate_tool.py`

When the Agent uses the `delegate_task` tool:

```python
def resolve_delegation_provider(configured_provider: str, ...) -> Dict:
    """Resolve the provider and credentials used by the sub-agent."""
    # Support configuring a different provider for the sub-agent
    # e.g., parent agent uses Nous, sub-agent uses OpenRouter + cheap model
```

The `delegation` section in `config.yaml` can configure:
- Provider used by sub-agent
- Model used by sub-agent
- Toolset available to sub-agent

---

## 7. Platform Adapter Routing

### 7.1 Architecture

**File**: `gateway/platforms/` directory

Each message is tagged with a `SessionSource` identifying the originating platform:

```python
@dataclass
class SessionSource:
    platform: Platform       # telegram, discord, slack, feishu, ...
    chat_id: str
    user_id: str
    thread_id: Optional[str] = None
    chat_type: str = "dm"   # "dm" | "group"
```

### 7.2 Platform Adapter Registration

**File**: `gateway/run.py`

```python
class GatewayRunner:
    adapters: Dict[Platform, BasePlatformAdapter] = {}

    # Register all enabled platform adapters at startup
    # Dynamically add/remove during runtime (hot-swap)
    # Message → _handle_message() → check source.platform → corresponding adapter
```


---

## 8. Hook System — Extensible Event Routing

**File**: `gateway/hooks.py` — `HookRegistry` class

Event-driven extensible routing:

```python
# Supported event types:
gateway:startup     # Gateway process started
session:start       # New session created (first message)
session:end         # Session ended (/new or /reset)
session:reset       # Session reset completed
agent:start         # Agent began processing message
agent:step          # Each iteration of the tool-call loop
agent:end           # Agent finished processing
command:*           # Any slash command executed (wildcard)
```

**Wildcard matching**: Handlers can register `"command:*"` to listen to all command events.

> [!NOTE]
> **Wildcard Routing is Essentially Prefix Matching, Upgradable to a Trie**
>
> `"command:*"` is algorithmically equivalent to string prefix matching, a classic use case for Tries or Aho-Corasick automata:
> - **Current implementation**: Iterate through all registered patterns for prefix comparison
> - **Scaling bottleneck**: O(n) iteration becomes a bottleneck when hook count reaches hundreds
> - **Upgrade path**: Prefix indexing based on Trie reduces matching to O(m) (m = command name length)
>
> At current scale (a few to a few dozen hooks), iteration is sufficient — this is a "good enough but reserves scalability" design boundary.

---

## 9. CLI-Side Routing

**File**: `cli.py` (main CLI REPL)

CLI-side routing is relatively simple, providing command completion via prompt_toolkit's `SlashCommandCompleter`:

```python
# CLI input → starts with `/` → go through command routing
# Otherwise → send to LLM normally
```

---

## 10. Summary: Hermes Intent Routing Overview

```
User message
    │
    ▼
┌─────────────────────────────┐
│  Message arrival layer      │
│  gateway/run.py:_handle_message() │
│  acp_adapter/server.py     │
└──────────┬──────────────────┘
           │
           ▼ Check `text.startswith("/")`
    ┌──────┴──────────────────┐
    │  Command detection      │
    │  (MessageEvent          │
    │  .get_command())        │
    └──────┬──────────────────┘
           │ Is a command
           ▼
    ┌─────────────────────────────────┐
    │  Command dispatch layer         │
    │  GATEWAY_KNOWN_COMMANDS match   │
    │  resolve_command() alias       │
    │  resolution                     │
    └─────┬───────────────────────────┘
          │
    ┌─────┼─────────────────┬──────────────┬─────────────┬─────────────┐
    │     │                 │              │             │             │
    ▼     ▼                 ▼              ▼             ▼             ▼
/btw    /background    Skill cmd      Quick cmd     Plugin cmd
(ephemeral  (bg task)   build_skill   exec/alias   get_plugin_
 LLM call)            invocation    direct exec  command_handler
           │           _message()              │
           │              │                     ▼
           ▼              ▼              Hand to LLM
        Normal user message (sent to AIAgent.run_conversation())
                    │
                    ▼
         ┌──────────────────────┐
         │  AIAgent tool-call   │
         │  loop                │
         │  model_tools decides │
         │  which tool to call  │
         │  (LLM autonomous)    │
         └──────────────────────┘
                    │
         ┌──────────┼──────────────┐
         ▼          ▼              ▼
    delegate_task  cron_job    send_message
    (sub-agent   (output       (platform
     routing)     delivery)     delivery)
```

### Key Design Principles

1. **Command routing takes priority over LLM**: All messages starting with `/` go through command routing first; only if no match is found are they sent to the LLM
2. **Unified alias resolution**: All aliases are normalized to canonical names before dispatch
3. **Layered decisions**: Model routing at the message layer, delivery routing at the output layer, platform routing at the access layer
4. **No ML intent recognition**: All routing decisions are based on string matching or keyword rules, no trained models
5. **Extensibility**: Supports dynamic extension through the Hook system, `register_plugin_command()`, and Skill mechanism
6. **Zero ambiguity handling**: Aliases point to only one canonical command; no polysemy exists

> [!NOTE]
> **The Implicit Philosophy of Hermes Routing: Determinism Over Flexibility, Latency Over Capability**
>
> After a comprehensive review of this document, an algorithmic philosophy emerges that permeates the entire system: Hermes's routing system is fundamentally a **rule-first, LLM-as-fallback layered decision tree**, not the "LLM-as-brain unified scheduler" approach popular in the industry today. This has several deep algorithmic implications:
>
> - **Zero-hallucination routing**: Rule-based routing never "gets it wrong" — given the same input, it guarantees the same output. LLM-based routing (if present) is inherently non-deterministic; the same input may yield different routing decisions due to temperature or context. In mission-critical systems, this is a reliability difference that cannot be ignored.
> - **Active latency budget management**: Each additional LLM call adds 100–500ms to p99 latency. Rule-based routing latencies are in the microsecond (dict lookup) to millisecond (I/O) range, a 2–3 order of magnitude difference. In multi-tenant Gateway scenarios, rule-first directly determines the system's capacity ceiling.
> - **Cost predictability**: LLM calls carry token costs that grow non-linearly with traffic (context expansion in long conversations). Rule-based routing costs are fixed infrastructure costs, easier to plan capacity and control budgets around.
>
> This design is reasonable and efficient under the current constraints of a fixed command set (hundreds of slash commands) and fixed platform set (a dozen messaging platforms). However, if future requirements call for **user-defined natural language intents** (e.g., "analyze my recent logs for me"), this system will face fundamental scaling challenges: keyword rules will explode combinatorially, O(n) rule maintenance becomes unsustainable — at that point, introducing a lightweight embedding-based classifier for natural language intent matching becomes almost inevitable.

> [!NOTE]
> **Comparison with Industry Mainstream Approaches: Rule Systems vs. LLM Routing Trade-offs**
>
> **Core insight**: The choice of routing strategy depends on whether the input space is closed or open.
>
> | Input Space | Optimal Solution | Reason |
> |-------------|-----------------|-------|
> | Closed (slash commands, finite and enumerable) | Rule routing | Provable correctness + O(1) complexity |
> | Open (natural language intents) | LLM routing | Cannot enumerate; requires semantic understanding |
>
> Hermes's slash command space is closed, and the designers made the correct technical choice. LLM-native routing (LangChain Agent Router, OpenAI Function Calling) suits open-domain scenarios, but in a closed space it is "using a sledgehammer to crack a nut."

