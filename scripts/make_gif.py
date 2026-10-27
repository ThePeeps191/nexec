"""README GIF: the salary-table heatmap swept through every layer, with the undefended and defended answers underneath.

    uv run python scripts/make_gif.py          # Llama-3.2-3B-Instruct, writes docs/assets/salary_layers.gif
"""

import sys
import tempfile
import textwrap
from pathlib import Path

from _common import LLAMA, ROOT
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(ROOT))
from evals.cases import salary_conversation  # noqa: E402
from nexec import NX  # noqa: E402
from nexec.viz import to_png, token_scores  # noqa: E402

WIDTH, PAD, FONT_PX, MAX_LINES = 1000, 10, 17, 3
FRAME_MS, HOLD_MS = 260, 2600


def _caption(answers: dict[str, str], font) -> Image.Image:
    lines = []
    for label, text in answers.items():
        body = textwrap.wrap(" ".join(text.split()), width=100)
        if len(body) > MAX_LINES:  # first line, then the answer's last line: that's where the result is
            last = [ln for ln in text.splitlines() if ln.strip()][-1]
            body = body[:1] + textwrap.wrap("… " + " ".join(last.split()), width=100)[:MAX_LINES - 1]
        lines.append((label, body))
    h = PAD + sum((1 + len(b)) * (FONT_PX + 6) + 8 for _, b in lines) + PAD
    img = Image.new("RGB", (WIDTH, h), "white")
    d = ImageDraw.Draw(img)
    bold = ImageFont.truetype(font_manager.findfont("DejaVu Sans:bold"), FONT_PX)
    y = PAD
    for label, body in lines:
        d.text((PAD, y), label, font=bold, fill="#222")
        y += FONT_PX + 6
        for line in body:
            d.text((PAD, y), line, font=font, fill="#333")
            y += FONT_PX + 6
        y += 8
    return img


def main() -> Path:
    nx = NX.from_pretrained(LLAMA)
    conv = salary_conversation("md_before")
    gen = dict(max_new_tokens=160)
    answers = {"Undefended answer:": nx.generate(conv, defend=False, **gen)[0],
               "With NXDirection (nexec, table marked untrusted):": nx.generate(conv, **gen)[0]}
    scores = token_scores(nx, conv, title="Salary table: Llama-3.2-3B-Instruct")
    font = ImageFont.truetype(font_manager.findfont("DejaVu Sans"), FONT_PX)
    cap = _caption(answers, font)
    frames, durations = [], []
    with tempfile.TemporaryDirectory() as tmp:
        for layer in range(scores.before.shape[0]):
            im = Image.open(to_png(scores, Path(tmp) / f"{layer}.png", layer=layer)).convert("RGB")
            im = im.resize((WIDTH, round(im.height * WIDTH / im.width)), Image.LANCZOS)
            frame = Image.new("RGB", (WIDTH, im.height + cap.height), "white")
            frame.paste(im, (0, 0))
            frame.paste(cap, (0, im.height))
            frames.append(frame.quantize(colors=128, method=Image.Quantize.MEDIANCUT))
            durations.append(HOLD_MS if layer == scores.default_layer else FRAME_MS)
    out = ROOT / "docs" / "assets" / "salary_layers.gif"
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)
    for k, v in answers.items():
        print(k, v)
    print(out, f"{out.stat().st_size / 1e6:.1f} MB")
    return out


if __name__ == "__main__":
    main()
