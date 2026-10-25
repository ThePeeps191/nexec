"""Render chat prompts with provenance: which tokens came from untrusted parts."""

from dataclasses import dataclass

from corpus import Example

# Llama-3.x templates stamp "Today Date"; pin it to the template's own default so renders are stable.
DATE_STRING = "26 Jul 2024"


@dataclass
class Rendered:
    input_ids: list[int]
    text: str
    part_char_ranges: list[tuple[int, int]]
    untrusted_mask: list[bool]
    offsets: list[tuple[int, int]]


def _supports_system(tok) -> bool:
    try:
        tok.apply_chat_template([{"role": "system", "content": "s"}, {"role": "user", "content": "u"}], tokenize=False)
        return True
    except Exception:
        return False


def _sentinel(i: int) -> str:
    return f"NXPART{i}"


def render(tok, parts: list[tuple[str, bool]], system: str | None = None) -> Rendered:
    parts = list(parts)
    messages = []
    if system is not None:
        if _supports_system(tok):
            messages.append({"role": "system", "content": system})
        else:
            parts.insert(0, (system + "\n\n", False))
    messages.append({"role": "user", "content": "".join(_sentinel(i) for i in range(len(parts)))})
    templated = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, date_string=DATE_STRING)

    # Substitute sentinels left to right, recording where each part's real text lands.
    text, ranges, cursor = "", [], 0
    for i, (part, _) in enumerate(parts):
        at = templated.index(_sentinel(i), cursor)
        text += templated[cursor:at]
        ranges.append((len(text), len(text) + len(part)))
        text += part
        cursor = at + len(_sentinel(i))
    text += templated[cursor:]

    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = [tuple(o) for o in enc["offset_mapping"]]
    untrusted = [r for r, (_, flag) in zip(ranges, parts) if flag]
    mask = [any(s < b and e > a for a, b in untrusted) for s, e in offsets]
    return Rendered(enc["input_ids"], text, ranges, mask, offsets)


def token_labels(tok, ex: Example, prefix: str = "") -> tuple[list[int], list[str | None]]:
    parts = ([(prefix, False)] if prefix else []) + [(ex.text, False)]
    r = render(tok, parts)
    base = r.part_char_ranges[-1][0]
    labels = []
    for s, e in r.offsets:
        # Byte-level BPE folds the preceding space into a token; judge the token by its visible text.
        while s < e and r.text[s].isspace():
            s += 1
        label = None
        if s < e:
            for span in ex.spans:
                if base + span.start <= s and e <= base + span.end:
                    label = span.label
                    break
        labels.append(label)
    return r.input_ids, labels
