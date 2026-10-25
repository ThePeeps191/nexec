"""Token provenance: render chat messages and mark which tokens came from untrusted sources.

A message's content is a string, an :class:`Untrusted`, or a list mixing both. ``role="tool"`` messages and
messages with ``"untrusted": True`` are untrusted as a whole. The result of :func:`render` carries an
``untrusted_mask`` aligned to its ``input_ids``; :func:`encode` batches several conversations with left padding.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import torch

# Llama-3.x templates stamp "Today Date"; pin it to the template's own default so renders are byte-stable.
DATE_STRING = "26 Jul 2024"


class TemplateError(ValueError):
    """The chat template rejected the conversation, or its output could not be aligned to the message content."""


@dataclass(frozen=True)
class Untrusted:
    text: str


def untrusted(text: str) -> Untrusted:
    """Mark a piece of message content as untrusted data (tool output, retrieved document, email, web page)."""
    return Untrusted(text)


@dataclass
class Rendered:
    input_ids: list[int]
    text: str
    untrusted_mask: list[bool]
    offsets: list[tuple[int, int]]
    untrusted_char_ranges: list[tuple[int, int]]
    part_char_ranges: list[list[tuple[int, int]]]  # per rendered message, per content part


@dataclass
class Encoded:
    input_ids: torch.Tensor       # [B, T] long, left-padded
    attention_mask: torch.Tensor  # [B, T] long
    untrusted_mask: torch.Tensor  # [B, T] bool
    renders: list[Rendered]


def _parts(msg: dict) -> list[tuple[str, bool]]:
    whole = bool(msg.get("untrusted", msg["role"] == "tool"))
    content = msg.get("content", "")
    if isinstance(content, Untrusted):
        return [(content.text, True)]
    if isinstance(content, str):
        return [(content, whole)]
    out = []
    for item in content:
        if isinstance(item, Untrusted):
            out.append((item.text, True))
        elif isinstance(item, str):
            out.append((item, whole))
        else:
            raise TypeError(f"message content items must be str or Untrusted, got {type(item).__name__}")
    return out


def _apply(tok, messages, add_generation_prompt, template_kwargs) -> str:
    try:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt,
                                       date_string=DATE_STRING, **(template_kwargs or {}))
    except Exception as e:  # jinja raise_exception surfaces as a TemplateError from jinja2
        raise TemplateError(str(e)) from e


def _supports_system(tok) -> bool:
    try:
        _apply(tok, [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}], False, None)
        return True
    except TemplateError:
        return False


def _normalize(tok, messages: list[dict]) -> list[tuple[str, list[tuple[str, bool]]]]:
    msgs = [(m["role"], _parts(m)) for m in messages]
    if msgs and msgs[0][0] == "system" and not _supports_system(tok):
        sys_parts = msgs[0][1] + [("\n\n", False)]
        rest = msgs[1:]
        if not rest or rest[0][0] != "user":
            raise TemplateError("template has no system role and the system message is not followed by a user turn")
        msgs = [("user", sys_parts + rest[0][1])] + rest[1:]
    return msgs


def _sentinel(i: int) -> str:
    return f"NX{i}"


def _ranges(parts: list[str], encode_part) -> list[tuple[int, int]]:
    out, pos = [], 0
    for p in parts:
        e = encode_part(p)
        out.append((pos, pos + len(e)))
        pos += len(e)
    return out


def render(tok, messages: list[dict], *, add_generation_prompt: bool = True,
           template_kwargs: dict | None = None) -> Rendered:
    msgs = _normalize(tok, messages)
    real = _apply(tok, [{"role": r, "content": "".join(t for t, _ in p)} for r, p in msgs], add_generation_prompt,
                  template_kwargs)
    skeleton = _apply(tok, [{"role": r, "content": _sentinel(i)} for i, (r, _) in enumerate(msgs)],
                      add_generation_prompt, template_kwargs)

    # Split the skeleton into the template text around each message, noting JSON-quoted content (tojson).
    chunks, json_mode, cursor = [], [], 0
    for i in range(len(msgs)):
        s = _sentinel(i)
        at = skeleton.find(s, cursor)
        if at < 0:
            raise TemplateError(f"template dropped the content of message {i}")
        quoted = skeleton[at - 1:at] == '"' and skeleton[at + len(s):at + len(s) + 1] == '"'
        chunks.append(skeleton[cursor:at - 1] if quoted else skeleton[cursor:at])
        json_mode.append(quoted)
        cursor = at + len(s) + (1 if quoted else 0)
    tail = skeleton[cursor:]

    # Walk the real render: template chunk, then the message content in whichever encoding the template used.
    pos, part_ranges, untrusted_ranges = 0, [], []
    for i, (_, parts) in enumerate(msgs):
        if not real.startswith(chunks[i], pos):
            raise TemplateError(f"template output diverges before message {i}")
        pos += len(chunks[i])
        texts = [t for t, _ in parts]
        joined = "".join(texts)
        nxt = chunks[i + 1] if i + 1 < len(msgs) else tail
        if json_mode[i]:
            enc = json.dumps(joined, ensure_ascii=False)
            local = [(a + 1, b + 1) for a, b in _ranges(texts, lambda p: json.dumps(p, ensure_ascii=False)[1:-1])]
        elif real.startswith(joined + nxt, pos):
            enc, local = joined, _ranges(texts, lambda p: p)
        else:
            enc = joined.strip()
            lead = len(joined) - len(joined.lstrip())
            local = [(min(max(a - lead, 0), len(enc)), min(max(b - lead, 0), len(enc)))
                     for a, b in _ranges(texts, lambda p: p)]
        if not real.startswith(enc + nxt, pos):
            raise TemplateError(f"content of message {i} not found verbatim in the rendered template")
        abs_ranges = [(pos + a, pos + b) for a, b in local]
        part_ranges.append(abs_ranges)
        untrusted_ranges += [r for r, (_, flag) in zip(abs_ranges, parts, strict=True) if flag and r[1] > r[0]]
        pos += len(enc)
    if real[pos:] != tail:
        raise TemplateError("template output diverges after the last message")

    enc_tok = tok(real, add_special_tokens=False, return_offsets_mapping=True)
    offsets = [tuple(o) for o in enc_tok["offset_mapping"]]
    mask = [any(s < b and e > a for a, b in untrusted_ranges) for s, e in offsets]
    return Rendered(list(enc_tok["input_ids"]), real, mask, offsets, untrusted_ranges, part_ranges)


def encode(tok, conversations: list[list[dict]], **render_kwargs) -> Encoded:
    """Render several conversations and left-pad them into one batch."""
    renders = [render(tok, conv, **render_kwargs) for conv in conversations]
    T = max(len(r.input_ids) for r in renders)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ids = torch.full((len(renders), T), pad, dtype=torch.long)
    att = torch.zeros((len(renders), T), dtype=torch.long)
    msk = torch.zeros((len(renders), T), dtype=torch.bool)
    for i, r in enumerate(renders):
        n = len(r.input_ids)
        ids[i, T - n:] = torch.tensor(r.input_ids)
        att[i, T - n:] = 1
        msk[i, T - n:] = torch.tensor(r.untrusted_mask, dtype=torch.bool)
    return Encoded(ids, att, msk, renders)


def mask_from_token_spans(n_tokens: int, spans: list[tuple[int, int]]) -> torch.Tensor:
    mask = torch.zeros(n_tokens, dtype=torch.bool)
    for a, b in spans:
        if not 0 <= a <= b <= n_tokens:
            raise ValueError(f"token span {(a, b)} out of range for {n_tokens} tokens")
        mask[a:b] = True
    return mask


def mask_from_char_spans(r: Rendered, spans: list[tuple[int, int]]) -> torch.Tensor:
    return torch.tensor([any(s < b and e > a for a, b in spans) for s, e in r.offsets], dtype=torch.bool)
