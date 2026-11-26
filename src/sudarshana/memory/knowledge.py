"""Knowledge layer: generated index.md + lint.md."""
import os

from sudarshana.core.config import VOLUME_PATH

# --- knowledge layer: generated index.md + lint.md (schema.md v2) -------------
# The compile pass writes pages; code owns the two derived files so they can't
# drift: index.md (one line per page, injected into every model call by the
# memory middleware) and lint.md (pages that break the schema, read by the
# compile). Rebuilt at the end of any turn in which a page changed.
KNOWLEDGE_DIR = os.path.join(VOLUME_PATH, "memory", "knowledge")
KNOWLEDGE_TYPES = ("concepts", "system", "built", "lessons")
KNOWLEDGE_PAGE_MAX_BYTES = 1024
KNOWLEDGE_TITLE_MAX_CHARS = 120
KNOWLEDGE_RELATED_MAX = 5
KNOWLEDGE_INDEX_HEADER = "# Knowledge index"
# Not a rule for the agent — a signal for us: past this the index costs too
# much to inject whole, and it's time to build per-turn retrieval instead.
# Raised 3000 -> 5000 on 2026-11-10 at Rinkesh's request: he wants the index
# to grow further before we decide retrieval is needed, so the signal should
# fire on real cost, not on a line we picked when the layer was young.
KNOWLEDGE_INDEX_SEARCH_THRESHOLD_TOKENS = 5000


def _parse_knowledge_page(path: str) -> dict:
    """Pull the schema-v2 fields out of one page. Tolerant of v1 pages: any
    field that isn't there comes back empty and shows up in lint.md."""
    import re

    with open(path, encoding="utf-8") as f:
        text = f.read()
    page = {"title": "", "evidence": "", "related": [], "added": "", "superseded_by": ""}
    for line in text.splitlines():
        s = line.strip()
        if not page["title"] and s.startswith("# "):
            page["title"] = s[2:].strip()
        elif s.startswith("Evidence:"):
            page["evidence"] = s[len("Evidence:"):].strip()
        elif s.startswith("Related:"):
            page["related"] = [r.strip() for r in s[len("Related:"):].split(",") if r.strip()]
        elif s.startswith("Added:"):
            page["added"] = s[len("Added:"):].strip()
        elif s.startswith("Superseded-by:"):
            m = re.match(r"([\w-]+)", s[len("Superseded-by:"):].strip())
            page["superseded_by"] = m.group(1) if m else ""
    page["bytes"] = len(text.encode("utf-8"))
    return page


def _maintain_knowledge(force: bool = False) -> None:
    """Regenerate index.md and lint.md if any page changed since index.md was
    last written. Deterministic, no model calls; never raises."""
    import glob
    from datetime import datetime, timezone

    try:
        index_path = os.path.join(KNOWLEDGE_DIR, "index.md")
        lint_path = os.path.join(KNOWLEDGE_DIR, "lint.md")
        page_paths = sorted(
            p for t in KNOWLEDGE_TYPES for p in glob.glob(os.path.join(KNOWLEDGE_DIR, t, "*.md"))
        )
        if not page_paths:
            return
        index_mtime = os.path.getmtime(index_path) if os.path.exists(index_path) else 0
        generated = False
        if index_mtime:
            with open(index_path, encoding="utf-8") as f:
                generated = f.readline().startswith(KNOWLEDGE_INDEX_HEADER)
        # Also rebuild when index.md wasn't written by this code (the v1
        # hand-written index, or a manual edit) — not only when a page changed.
        if not force and generated and max(os.path.getmtime(p) for p in page_paths) <= index_mtime:
            return

        pages = {}  # slug -> page
        issues = []
        for p in page_paths:
            rel = os.path.relpath(p, KNOWLEDGE_DIR).replace(os.sep, "/")
            slug = os.path.splitext(os.path.basename(p))[0]
            page = _parse_knowledge_page(p)
            page["rel"], page["type"] = rel, rel.split("/")[0]
            if slug in pages:
                issues.append(f"{rel}: slug also used by {pages[slug]['rel']}")
            pages[slug] = page

        inbound = {slug: 0 for slug in pages}
        for slug, page in pages.items():
            rel = page["rel"]
            if not page["title"]:
                issues.append(f"{rel}: no '# ' title line")
            elif len(page["title"]) > KNOWLEDGE_TITLE_MAX_CHARS:
                issues.append(f"{rel}: title {len(page['title'])} chars (limit {KNOWLEDGE_TITLE_MAX_CHARS})")
            if page["bytes"] > KNOWLEDGE_PAGE_MAX_BYTES:
                issues.append(f"{rel}: {page['bytes']} bytes (limit {KNOWLEDGE_PAGE_MAX_BYTES})")
            if not page["evidence"]:
                issues.append(f"{rel}: no 'Evidence:' line")
            if not page["added"]:
                issues.append(f"{rel}: no 'Added:' line")
            if len(page["related"]) > KNOWLEDGE_RELATED_MAX:
                issues.append(f"{rel}: {len(page['related'])} Related links (limit {KNOWLEDGE_RELATED_MAX})")
            for r in page["related"]:
                if r in pages:
                    inbound[r] += 1
                else:
                    issues.append(f"{rel}: Related -> '{r}' (no such page)")
            if page["superseded_by"] and page["superseded_by"] not in pages:
                issues.append(f"{rel}: Superseded-by -> '{page['superseded_by']}' (no such page)")
        for slug, n in inbound.items():
            if n == 0 and not pages[slug]["superseded_by"]:
                issues.append(f"{pages[slug]['rel']}: orphan (no page lists it in Related)")

        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        live = {s: p for s, p in pages.items() if not p["superseded_by"]}
        body = []
        for t in KNOWLEDGE_TYPES:
            rows = [p for p in live.values() if p["type"] == t]
            if rows:
                body.append(f"\n## {t}")
                for p in sorted(rows, key=lambda p: p["rel"]):
                    title = p["title"] or "(untitled)"
                    if len(title) > KNOWLEDGE_TITLE_MAX_CHARS:
                        title = title[: KNOWLEDGE_TITLE_MAX_CHARS - 1] + "…"
                    body.append(f"- {p['rel']} — {title}")
        superseded = [p for p in pages.values() if p["superseded_by"]]
        if superseded:
            body.append("\n## superseded (history only)")
            for p in sorted(superseded, key=lambda p: p["rel"]):
                body.append(f"- {p['rel']} → {p['superseded_by']}")
        body_text = "\n".join(body)
        tokens = len(body_text) // 4
        header = (
            f"{KNOWLEDGE_INDEX_HEADER} — {len(live)} pages · ~{tokens} tokens · generated {now}\n"
            "Generated by code from the pages; never edit this file. "
            "read_file a path for the page body and its Evidence."
        )
        with open(index_path, "w", encoding="utf-8") as f:
            f.write(header + "\n" + body_text + "\n")

        lint = [f"# Knowledge lint — {now} — {len(issues)} issue(s)",
                "Generated by code against schema.md; never edit. Fix these as the compile touches each page.", ""]
        lint += [f"- {i}" for i in sorted(issues)] or ["No issues."]
        with open(lint_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lint) + "\n")

        print(f"[knowledge] index: {len(live)} pages, ~{tokens} tokens; lint: {len(issues)} issue(s)")
        if tokens > KNOWLEDGE_INDEX_SEARCH_THRESHOLD_TOKENS:
            print(f"[knowledge] index is past {KNOWLEDGE_INDEX_SEARCH_THRESHOLD_TOKENS} tokens — time to build retrieval")
    except Exception as e:
        print(f"[knowledge] maintenance failed: {e!r}")
