"""The small core pieces PR 3 added: the dedup helper and the owner-channel prompt.

Why: Telegram's dedup now goes through one helper any channel can use, and
the prompt names the owner's chat from the notifier instead of hard-coding
"Telegram". These pin that the helper keeps a bounded window per scope, and
that the prompt has exactly the two channel slots it had Telegram mentions.
"""
from sudarshana.core import dedup, prompt
from sudarshana.infra.storage.local.stores import DictKV


def test_seen_is_per_scope_and_bounded():
    kv = DictKV()
    assert not dedup.is_seen(kv, "seen:webhook", 1)
    dedup.mark_seen(kv, "seen:webhook", 1, window=3)
    assert dedup.is_seen(kv, "seen:webhook", 1)
    assert not dedup.is_seen(kv, "seen:run", 1)
    for i in range(2, 6):
        dedup.mark_seen(kv, "seen:webhook", i, window=3)
    assert kv.get("seen:webhook") == [3, 4, 5]


def test_marking_twice_keeps_one_copy_at_the_end():
    kv = DictKV()
    for i in (1, 2, 1):
        dedup.mark_seen(kv, "s", i, window=10)
    assert kv.get("s") == [2, 1]


def test_prompt_names_the_owner_channel_twice():
    assert prompt.SYSTEM_PROMPT.count("{owner_channel}") == 2
    rendered = prompt.SYSTEM_PROMPT.replace("{owner_channel}", "Telegram")
    assert "He gets it on Telegram as" in rendered
    assert "a Telegram webhook (this chat, private to Rinkesh)" in rendered
