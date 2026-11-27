"""Building the agent and running one turn."""
import os
from dataclasses import dataclass
from pathlib import Path

from sudarshana.core.config import VOLUME_PATH
from sudarshana.core.messages import _final_message, _format_blurb, _timestamp
from sudarshana.core.prompt import SYSTEM_PROMPT
from sudarshana.core.tracing import _build_call_logger, _build_timing_handler
from sudarshana.integrations.a2a import _a2a_log, _a2a_screen
from sudarshana.integrations.broker import broker_register, broker_request
from sudarshana.memory.asks import ASK_DEFAULT_DAYS, _ask_create, _ask_resolve, _asks_context
from sudarshana.memory.inbox import _inbox_intake_context, _inbox_set_status
from sudarshana.memory.knowledge import _maintain_knowledge
from sudarshana.memory.report import _report_save
from sudarshana.memory.roadmap import (
    _initiative_create,
    _initiative_update,
    _roadmap_context,
    _roadmap_view,
    _task_add,
    _task_update,
)
from sudarshana.safety.tool_screen import (
    SHELL_WITHHELD_SECRETS,
    TOOL_SCREEN_ENABLED,
    _build_tool_screen,
)

# --- Conversation thread: LangGraph checkpointer -----------------------------
# Every turn — Telegram message or hourly wake-up — runs on one
# checkpointed thread, so a reply ("yes, do it") sees the conversation before
# it, including the wake-up report it answers. The thread keeps everything —
# tool calls and results too; deepagents' built-in SummarizationMiddleware
# condenses the oldest part when it nears the context limit.
#
# Storage is a SQLite file on the Volume (SqliteSaver, opened in setup()),
# saved by each turn's volume.commit(). DELETE journal, not SqliteSaver's WAL
# default — WAL's -wal/-shm side files don't survive a network volume (same
# choice as the visitor-inbox store). Overlapping turns in two containers (a
# Telegram message during a wake-up) race on the file: last commit wins, and
# the other turn's exchange drops out of the thread. Acceptable for one user.
CHECKPOINT_DB_PATH = os.path.join(VOLUME_PATH, "checkpoints.db")
CONVERSATION_THREAD = {"configurable": {"thread_id": "sudarshana"}}


# Max graph steps per turn; also quoted in the Telegram message when hit.
# ~4 steps per model call. 200 (~50 calls) was hit by ~1 in 3 turns on
# 11-13..11-14, mostly while writing notes after the work landed; those turns
# were 97% unique tool calls, i.e. real work, not loops. 300 (~75 calls)
# still stops a runaway.
RECURSION_LIMIT = 300
# Returned by _invoke() when a turn hit the step cap instead of finishing, so
# callers can close the turn with an honest outcome rather than "done".
STEP_LIMIT = "step-limit"


class Agent:
    """A built agent: the LangGraph graph, plus what its tools reach per turn."""

    def __init__(self, deps):
        self.deps = deps
        self.graph = None
        self.a2a = None
        # The turn report_status writes to. Set per turn, not read from the
        # status store's `current`: inputs in a container run one at a time,
        # while `current` is shared across containers and names whichever turn
        # started last.
        self.turn_key = None


@dataclass
class TurnResult:
    reply: str
    outcome: str  # "done" or STEP_LIMIT


def build_agent(deps) -> Agent:
    """Build the agent once per container; every turn reuses it."""
    agent = Agent(deps)
    from deepagents import create_deep_agent
    from deepagents.backends import CompositeBackend, LocalShellBackend
    from deepagents.backends.filesystem import FilesystemBackend
    from deepagents.middleware.memory import MemoryMiddleware
    from deepagents.middleware.skills import SkillsMiddleware
    from deepagents.middleware.summarization import SummarizationMiddleware
    from langchain_core.tools import tool
    from langchain_openai import ChatOpenAI

    # Appends state.md and the knowledge index to the compiled system
    # message. The small custom template keeps per-call cost low (the
    # default MEMORY_SYSTEM_PROMPT is ~1.6k tokens, geared to AGENTS.md).
    #
    # Stock MemoryMiddleware loads once per thread, which on the
    # checkpointed thread would pin the first state.md forever, so reload
    # every turn. Upstream bug: langchain-ai/deepagents#6122 — drop this
    # subclass once a release fixes it.
    class FreshMemoryMiddleware(MemoryMiddleware):
        def before_agent(self, state, runtime, config):
            state = {k: v for k, v in state.items() if k != "memory_contents"}
            return super().before_agent(state, runtime, config)

    memory_middleware = FreshMemoryMiddleware(
        backend=FilesystemBackend(root_dir="/"),
        sources=[
            f"{VOLUME_PATH}/memory/state.md",
            f"{VOLUME_PATH}/memory/knowledge/index.md",
        ],
        add_cache_control=False,  # Anthropic-only
        system_prompt=(
            "--- where I am right now (from /data/memory/state.md, "
            "refreshed each cycle; the canonical files "
            "roadmap store/docs/VISION/logs always win on disagreement) "
            "---\n{agent_memory}"
        ),
    )

    # Skills library at /data/skills/ (README.md there documents the
    # format). Same load-once-per-thread issue as memory, so rescan every
    # turn. Upstream: langchain-ai/deepagents#5416 — drop once fixed.
    class FreshSkillsMiddleware(SkillsMiddleware):
        def before_agent(self, state, runtime, config):
            state = {k: v for k, v in state.items() if k != "skills_metadata"}
            return super().before_agent(state, runtime, config)

    # Only when a skill actually exists. The middleware's own instructions
    # ("no skills available yet", plus its how-to) ride on every model call
    # and cost ~480 tokens to say the library is empty; /data/skills holds
    # only its README. The scan runs once per container start, so a newly
    # written skill is picked up at the next cold start, not mid-container.
    skills_dir = Path(f"{VOLUME_PATH}/skills")
    has_skills = any(skills_dir.glob("*/SKILL.md"))
    skills_middleware = (
        FreshSkillsMiddleware(
            backend=FilesystemBackend(root_dir="/"),
            sources=[f"{VOLUME_PATH}/skills/"],
        )
        if has_skills
        else None
    )

    @tool
    def search_web(query: str) -> str:
        """Search the public web (DuckDuckGo). Use when you need current
        or external information that isn't in your files: news, docs,
        prices, facts. Free, no API key. Returns a few top results as
        plain text: title, URL, and a short snippet."""
        from ddgs import DDGS
        from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException

        try:
            with DDGS() as ddgs:
                raw = ddgs.text(query, max_results=5) or []
        except RatelimitException:
            return "(no results: DuckDuckGo rate-limited this search — try again shortly)"
        except TimeoutException:
            return "(no results: DuckDuckGo search timed out — try again shortly)"
        except DDGSException as e:
            # ddgs raises (not returns []) when a query matches nothing;
            # "search failed" made the agent misread that as throttling.
            if "no results" in str(e).lower():
                return "(no results for this query — try different wording)"
            return f"(search failed: {e})"

        results = []
        for r in raw:
            title = r.get("title", "").strip()
            url = r.get("href", "").strip()
            body = r.get("body", "").strip()
            if title or url:
                results.append(f"- {title}\n  {url}")
                if body:
                    results[-1] += f"\n  {body}"
        return "\n\n".join(results) if results else "(no results)"

    search_tools = [search_web]

    # The model acts on the visitor inbox only through these tools and the
    # per-call intake note, never by editing /data/visitor-inbox.db.

    @tool
    def inbox_review() -> str:
        """Review the visitor-inbox queue: list PRIVATE received items
        awaiting your policy check, plus any already-public board items
        awaiting work. Nothing you type into the form is public until you
        approve it. Returns a compact summary (or that the inbox is idle)."""
        ctx = _inbox_intake_context()
        return (
            ctx
            if ctx
            else f"Visitor inbox idle: 0 pending, 0 active."
        )

    @tool
    def inbox_set_status(item_id: str, status: str, note: str = "", artifact_link: str = "", answer: str = "") -> str:
        """Advance one visitor-inbox item through its lifecycle. The ONLY
        call that makes an item public (approve received -> submitted) or
        retracts one from the public board (-> rejected, private terminal;
        or submitted -> in_progress -> completed to work it). Forward-only,
        validated transitions only. Use only after your own policy judgment.
        When COMPLETING an item (-> completed) write the full answer as its
        OWN long-form content entity via answer=<the detailed response>:
        paragraphs like a blog post but NOT a blog — the item row carries
        only the link to it (artifact_link -> answer id) and the public
        board's detail view serves the full text. artifact_link is optional
        and only for an external artifact. note is the private audit trail
        only.
        Returns a short result string."""
        res = _inbox_set_status(deps, item_id, status, note=note, artifact_link=artifact_link, answer=answer)
        if res.get("ok"):
            vis = "PUBLIC (on the site board)" if res.get("public") else "private"
            return f"ok: {item_id} -> {status} ({vis})"
        return f"failed: {res.get('error', 'unknown')}"

    inbox_tools = [inbox_review, inbox_set_status]

    @tool
    def ask_rinkesh(question: str, why: str, options: str = "", expires_days: int = ASK_DEFAULT_DAYS) -> str:
        """Ask Rinkesh for something only he can give — a decision, an
        approval, a merge, an action on his side. Sends him one Telegram
        message right away and tracks it as an open ask until you close it
        with resolve_ask or it lapses (expires_days, default 3, max 14).
        One question per ask, short enough to answer from a phone. why= is
        the one-line reason it's blocked on him; options= lists the choices
        if there are any (put your recommendation first). At most 3 asks
        can be open at once. Returns the ask id, or why it wasn't sent."""
        res = _ask_create(deps, question, why, options=options, expires_days=expires_days)
        if res.get("ok"):
            return f"sent: [{res['id']}], open until {res['expires']}"
        return f"not sent: {res['error']}"

    @tool
    def resolve_ask(ask_id: str, outcome: str, status: str = "answered") -> str:
        """Close an open ask to Rinkesh. status='answered' when he answered
        it; status='dropped' when you no longer need it (it got done
        another way, the work moved on, it was wrong to ask). outcome= is
        one line: what he said, or why it's no longer needed."""
        res = _ask_resolve(deps, ask_id, outcome, status=status)
        if res.get("ok"):
            return f"closed: [{res['id']}] {res['status']}"
        return f"failed: {res['error']}"

    @tool
    def report_status(doing: str, why: str, worked_on: list[str] | None = None,
                      next_step: str = "") -> str:
        """Tell Rinkesh where things stand. He reads this on /control at
        any time to see what you are doing and why. Call it ONCE, as the
        last thing in EVERY turn — hourly or his message — after the work
        and the log line. Each call replaces the last report.
        doing= one plain sentence: what this turn did, or what is under
        way. why= the reason in terms of what Rinkesh asked for (quote his
        words when you can), not your internal bookkeeping. worked_on=
        the roadmap items this turn worked on, each "initiative" or
        "initiative/task" (e.g. ["gate-axe/t2", "post-meta"]); [] for a
        turn that wasn't roadmap work (blog, inbox, a chat). Ids are
        checked against the roadmap; progress (status, done_when, tasks
        done/left) is filled in from it, so don't restate it. next_step=
        what the next turn will do, or "waiting on Rinkesh" when it is
        blocked on an open ask. Plain words; no file paths or commit
        hashes unless he needs them to act."""
        res = _report_save(deps, agent.turn_key, doing, why,
                           worked_on=worked_on, next_step=next_step)
        if res.get("ok"):
            return f"reported at {res['report']['reported_at']}"
        return f"not reported: {res['error']}"

    ask_tools = [ask_rinkesh, resolve_ask, report_status]

    # The roadmap: initiatives and their tasks. These tools are the only
    # writers of /data/memory/roadmap.json; the prose lives in each
    # initiative's doc (/data/actions/<id>.md).

    def _rm_fail(res: dict) -> str:
        return f"failed: {res.get('error', 'unknown')}"

    @tool
    def roadmap_view(initiative_id: str = "") -> str:
        """Read the roadmap. With no id: every initiative, one line each
        (status, priority, open tasks). With an id: that
        initiative in full — objective, done_when, doc, every task with
        its handoff and deliverables."""
        return _roadmap_view(initiative_id)

    @tool
    def create_initiative(id: str, title: str, objective: str, done_when: str,
                          asked_by: str = "", status: str = "proposed", priority: int = 0,
                          links: list[str] | None = None) -> str:
        """Add an initiative to the roadmap. id= short lower-kebab-case
        (e.g. "gate-axe"). objective= the outcome in one or two
        sentences; done_when= the finish line, checkable. asked_by=
        Rinkesh's words if he asked for it. status= 'proposed' (default)
        or 'active' — an initiative of your own stays proposed until he
        says yes. priority= lower runs first (0 default). links= extra
        /data paths or URLs. Creates the doc /data/actions/<id>.md if it
        doesn't exist; write your notes there. Add its tasks with add_task."""
        res = _initiative_create(deps, id, title, objective, done_when, asked_by=asked_by,
                                 status=status, priority=priority, links=links)
        if not res.get("ok"):
            return _rm_fail(res)
        it = res["initiative"]
        return f"created {it['id']} [{it['status']}], doc {it['doc']}"

    @tool
    def update_initiative(id: str, status: str = "", title: str = "", objective: str = "",
                          done_when: str = "", priority: int | None = None,
                          blocked_on: str = "", reason: str = "", outcome: str = "",
                          links: list[str] | None = None) -> str:
        """Change an initiative. You keep its status current yourself.
        Status moves allowed: proposed -> active | declined; active ->
        waiting (needs blocked_on=, e.g. 'ask 3f2a') | blocked (needs
        reason=) | done | declined; waiting -> active | blocked |
        declined; blocked -> active | declined. done/declined need
        outcome= and are final; the doc moves to /data/actions/archive/.
        links= replaces the list."""
        res = _initiative_update(deps, id, status=status, title=title, objective=objective,
                                 done_when=done_when, priority=priority,
                                 blocked_on=blocked_on, reason=reason,
                                 outcome=outcome, links=links)
        if not res.get("ok"):
            return _rm_fail(res)
        it = res["initiative"]
        return f"ok: {it['id']} [{it['status']}], doc {it['doc']}"

    @tool
    def add_task(initiative_id: str, title: str, done_when: str, position: int | None = None) -> str:
        """Add a task to an initiative: one deliverable that takes about
        1-3 cycles (bigger than that, split it). done_when= what proves it
        finished. position= 1-based place in the queue (default: last).
        Tasks run in queue order, one 'doing' at a time."""
        res = _task_add(deps, initiative_id, title, done_when, position=position)
        if not res.get("ok"):
            return _rm_fail(res)
        return f"added {res['initiative']}/{res['task']['id']}"

    @tool
    def update_task(initiative_id: str, task_id: str, status: str = "", title: str = "",
                    done_when: str = "", handoff: str = "", deliverables: list[str] | None = None,
                    blocked_on: str = "", reason: str = "") -> str:
        """Change a task. Status moves allowed: todo -> doing | dropped;
        doing -> todo | waiting (needs blocked_on=) | blocked (needs
        reason=) | done | dropped; waiting/blocked -> doing | dropped.
        Waiting needs blocked_on= (e.g. 'ask 3f2a'); move it back to
        doing yourself once that's settled. Only
        one task per initiative is 'doing', and only on an active
        initiative. done needs deliverables= (URLs or existing /data paths
        it produced) and a handoff=; dropped needs reason=. handoff= one or
        two lines the NEXT cycle reads first — set it every cycle you work
        the task, even if it's not done. deliverables= are added to the
        list."""
        res = _task_update(deps, initiative_id, task_id, status=status, title=title,
                           done_when=done_when, handoff=handoff, deliverables=deliverables,
                           blocked_on=blocked_on, reason=reason)
        if not res.get("ok"):
            return _rm_fail(res)
        t = res["task"]
        return f"ok: {res['initiative']}/{t['id']} [{t['status']}]"

    roadmap_tools = [roadmap_view, create_initiative, update_initiative, add_task, update_task]
    broker_tools = [tool(broker_register), tool(broker_request)]
    search_tools = [*search_tools, *inbox_tools, *ask_tools, *roadmap_tools, *broker_tools]

    # Production runs on OpenRouter (USE_OPENROUTER=1, OPENROUTER_MODEL /
    # OPENROUTER_API_KEY). The else branch targets the self-hosted Qwen
    # endpoint on Modal, which is stopped; it is kept only as a fallback.
    # Groq was tried and dropped: its 8K tokens/min cap is below this
    # agent's ~12K-token calls.
    if os.environ.get("USE_OPENROUTER"):
        # ChatOpenRouter, not ChatOpenAI: it keeps the model's reasoning
        # fields (ChatOpenAI drops them) and sends them back on later
        # calls, which DeepSeek requires whenever tools are present.
        from langchain_openrouter import ChatOpenRouter

        llm = ChatOpenRouter(
            model=os.environ["OPENROUTER_MODEL"],
            api_key=os.environ["OPENROUTER_API_KEY"],
            # Reasoning models spend max_tokens on thinking; 4096 ended
            # runs empty.
            max_tokens=32768,
            # Milliseconds here, unlike ChatOpenAI's seconds (10 minutes).
            timeout=600_000,
            # OpenRouter attribution (HTTP-Referer / X-Title). The
            # defaults are LangChain's, so usage showed as "langchain".
            # Same referer as the Jev calls.
            app_url="https://github.com/rinkesh2010rpp/sudarshana",
            app_title="sudarshana",
            # No provider pinning: pinning Fireworks cost ~$30/mo. If a
            # cheap host mangles tool calls, add a provider "ignore" /
            # "quantizations" filter rather than an order.
        )
    else:
        llm = ChatOpenAI(
            model=os.environ.get("LLM_MODEL", "qwen"),
            base_url=os.environ.get(
                "LLM_BASE_URL",
                "https://rinkesh2010rpp--llm-inference-vllmserver-serve.modal.run/v1",
            ),
            api_key=os.environ.get("LLM_API_KEY", "dummy"),
            max_tokens=32768,
            timeout=600,
        )
    # deepagents (>=0.7.22) refuses any call above 95% of max_input_tokens
    # minus max_tokens: 128K here → ~88K. DeepSeek V4 takes 1M, but
    # unpinned hosts may cap lower, so stay conservative. Summarization is
    # set explicitly at 70K below, not as a fraction of this limit (the
    # default 85% would sit above the ~88K budget and never get to run).
    llm.profile = {**(llm.profile or {}), "max_input_tokens": 128_000}

    # The shell is passed an explicit env = os.environ minus the named
    # model/telegram/webhook secrets (SHELL_WITHHELD_SECRETS); the system
    # vars and GITHUB_TOKEN stay. This only raises the bar — it stops a
    # naive `env`/`printenv` dump and accidental logging of those keys. It
    # is NOT isolation: the parent Python process still holds every secret
    # (the model client needs them), so a shell child can read them from
    # /proc/<ppid>/environ, and a secret added to .env later passes through
    # until it is named here. The real fix is running the shell in a
    # separate sandbox without the secrets (not done yet). virtual_mode=
    # False so file tools and the shell agree on paths (the default remaps
    # "/X" to "/data/X" for file tools only). The no-route CompositeBackend
    # only sets artifacts_root:
    # middleware writes compacted history and evicted tool results under
    # <artifacts_root>/, which defaults to "/" (container disk, lost on
    # restart). Everything else passes through to the shell backend.
    shell_env = {
        k: v for k, v in os.environ.items() if k not in SHELL_WITHHELD_SECRETS
    }
    backend = CompositeBackend(
        default=LocalShellBackend(
            root_dir=VOLUME_PATH,
            virtual_mode=False,
            env=shell_env,
            inherit_env=False,
        ),
        routes={},
        artifacts_root=VOLUME_PATH,
    )

    def summarization():
        # Replaces the built-in one by name. Compacts the thread at ~70K
        # tokens and keeps the last ~8K, as the old 82K profile did.
        return SummarizationMiddleware(
            llm,
            backend=backend,
            trigger=("tokens", 70_000),
            keep=("tokens", 8_000),
            # The class default (4000) shows the summary model only the
            # last 4K tokens; the built-in factory passes None (all of it).
            trim_tokens_to_summarize=None,
            truncate_args_settings={
                "trigger": ("tokens", 70_000),
                "keep": ("tokens", 8_000),
            },
        )

    checkpointer = deps.checkpoints.open()

    # deepagents no longer adds write_todos by default; the prompt relies on it.
    from deepagents.middleware.subagents import GENERAL_PURPOSE_SUBAGENT
    from langchain.agents.middleware import TodoListMiddleware

    from a2a_hotplug import A2AMiddleware

    # Registers add_agent / send_agent_task / check_agent_task itself (like
    # deepagents' FilesystemMiddleware). Middleware tools aren't handed to
    # the `task` subagent, which keeps outward A2A calls in the main agent.
    # Also the handle for the /agents controls in telegram_webhook.
    agent.a2a = A2AMiddleware(
        store=deps.a2a_kv, screen=_a2a_screen, notify=deps.owner_notifier.send, log=_a2a_log
    )

    main_middleware = [
        memory_middleware,
        *([skills_middleware] if skills_middleware else []),
        TodoListMiddleware(),
        agent.a2a,
        summarization(),
    ]
    # deepagents doesn't hand custom middleware to its auto-added
    # general-purpose subagent, which has search_web and execute too, so
    # declare it with its own. A declared subagent gets the main agent's
    # model and tools but not its middleware.
    subagent_middleware = [summarization(), TodoListMiddleware()]
    if TOOL_SCREEN_ENABLED:
        taster = _build_tool_screen()  # one instance, shared by both
        main_middleware.append(taster)
        subagent_middleware.append(taster)

    agent.graph = create_deep_agent(
        model=llm,
        system_prompt=SYSTEM_PROMPT.replace("{owner_channel}", deps.owner_notifier.channel),
        middleware=main_middleware,
        subagents=[{**GENERAL_PURPOSE_SUBAGENT, "middleware": subagent_middleware}],
        tools=search_tools,
        checkpointer=checkpointer,
        backend=backend,
    )
    return agent


def run_turn(agent: Agent, message: str, trigger: str, turn_key: str | None = None) -> TurnResult:
    """Run one turn and return its reply. Sending the reply is the caller's
    job: the interface the message came in on answers on that interface."""
    from langgraph.errors import GraphRecursionError

    # The turn this call belongs to, for report_status. An attribute on
    # this instance, not current._turn_key: inputs in this container run
    # one at a time, while `current` is shared across containers and
    # names whichever turn started last.
    agent.turn_key = turn_key

    invoke_input = {
        "messages": [
            # Current time goes in a fresh system message per call; baked
            # into the compiled system_prompt it would go stale. Kept as-is
            # until a better mechanism turns up.
            {
                "role": "system",
                "content": f"Current time: {_timestamp()}",
            },
            # Visitor inbox items, every cycle; adds nothing when idle.
            *([] if not (_icc := _inbox_intake_context()) else [
                {"role": "system", "content": f"Visitor inbox intake:\n{_icc}"}
            ]),
            # Open asks to Rinkesh, every turn; adds nothing when none.
            *([] if not (_acx := _asks_context(agent.deps)) else [
                {"role": "system", "content": _acx}
            ]),
            # The roadmap's focus initiative, current task and handoff.
            {"role": "system", "content": _roadmap_context()},
            {"role": "user", "content": message},
        ]
    }
    # The step cap catches a looping run; the timeout is the real backstop.
    cfg = {"callbacks": [_build_timing_handler(), _build_call_logger(trigger)], "recursion_limit": RECURSION_LIMIT, **CONVERSATION_THREAD}
    try:
        # Checkpoint once at turn end, not per step (~77 snapshots/turn);
        # we never resume mid-turn.
        result = agent.graph.invoke(invoke_input, config=cfg, durability="exit")
    except GraphRecursionError:
        # Don't let this crash the invocation — that sends nothing back.
        # Report and move on.
        # A distinct outcome, not "done": this turn did not finish, and
        # closing it as "done" hid exactly the failure the panel exists to
        # show. The callers record it as the `step-limit` outcome.
        return TurnResult(
            reply=f"[hit the {RECURSION_LIMIT}-step safety limit this cycle without finishing — "
            "stopping. Likely looping or over-scoped. No trace for this run.]",
            outcome=STEP_LIMIT,
        )
    finally:
        # Rebuild index.md / lint.md if the turn changed any knowledge
        # page — before the caller's storage commit, so they land with it.
        _maintain_knowledge()
    # Full trace of this turn to the modal logs for debugging (the thread
    # holds every earlier turn too, so start at this turn's message); only
    # the agent's final message is the reply.
    msgs = result.get("messages", [])
    start = max(
        (i for i, m in enumerate(msgs) if type(m).__name__ == "HumanMessage" and m.content == message),
        default=0,
    )
    print(_format_blurb(msgs[start:]))
    return TurnResult(reply=_final_message(msgs), outcome="done")
