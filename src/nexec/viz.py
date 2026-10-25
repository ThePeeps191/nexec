"""Per-token instruction-likeness heatmaps: h·d at every layer, before and after NXDirection.

``token_scores`` runs two forward passes over one prompt: undefended, and inside ``nx.protect`` (capture hooks are
registered after the NX hooks, so they see the edited residual). ``to_html`` writes a self-contained page with a layer
slider and a before/after toggle; ``to_png`` writes a static figure for READMEs (needs the ``nexec[viz]`` extra).
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class TokenScores:
    tokens: list[str]      # exact source text of each token (from offsets)
    untrusted: list[bool]
    content: list[bool]    # inside some message's content (vs chat-template/special tokens)
    message: list[int]     # index of the message whose content holds the token (-1 for template tokens)
    before: torch.Tensor   # [L, T] h.d, undefended pass
    after: torch.Tensor    # [L, T] h.d, defended pass (after the edit at hooked layers)
    bias: torch.Tensor     # [L] gate threshold b (zeros for the raw gate)
    hooked: list[int]
    default_layer: int
    title: str = ""


def default_layer(before: torch.Tensor, untrusted: list[bool], bias: torch.Tensor) -> int:
    """Layer in 0..L//2 where the undefended pass gives the gate the most to remove; L//4 if nothing fires."""
    L = before.shape[0]
    m = torch.tensor(untrusted, dtype=torch.bool)
    if not m.any():
        return L // 4
    mass = torch.relu(before[: L // 2 + 1, m] - bias[: L // 2 + 1, None]).sum(1)
    return int(mass.argmax()) if float(mass.max()) > 0 else L // 4


@torch.no_grad()
def _projections(model, ids: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    outs: dict[int, torch.Tensor] = {}
    layers = model.model.layers
    def grab(l):
        def hook(module, args, output):
            if not isinstance(output, torch.Tensor):
                raise TypeError(f"decoder layer {l} returned {type(output).__name__}; expected a tensor")
            outs[l] = output[0].float().cpu()  # batch row 0 (token_scores takes one conversation)
        return hook

    hs = [layer.register_forward_hook(grab(l)) for l, layer in enumerate(layers)]
    try:
        model(ids, use_cache=False)
    finally:
        for h in hs:
            h.remove()
    return torch.stack([outs[l] @ d[l].float() for l in range(len(layers))])


def token_scores(nx, conversation: list[dict], *, title: str = "") -> TokenScores:
    enc = nx.encode(conversation)
    if enc.input_ids.shape[0] != 1:
        raise ValueError("token_scores takes a single conversation")
    r = enc.renders[0]
    ids = enc.input_ids.to(nx.model.device)
    d = nx.directions.d
    before = _projections(nx.model, ids, d)
    with nx.protect(enc.untrusted_mask):
        after = _projections(nx.model, ids, d)
    message = [next((k for k, msg in enumerate(r.part_char_ranges) if any(s < b and e > a for a, b in msg)), -1)
               for s, e in r.offsets]
    content = [m >= 0 for m in message]
    bias = nx.directions.bias()
    bias = torch.zeros(d.shape[0]) if bias is None else bias.float()
    untrusted = enc.untrusted_mask[0].tolist()
    return TokenScores(tokens=[r.text[s:e] for s, e in r.offsets], untrusted=untrusted, content=content,
                       message=message,
                       before=before, after=after, bias=bias, hooked=list(nx.directions.config.layers),
                       default_layer=default_layer(before, untrusted, bias), title=title)


def _scales(s: TokenScores) -> list[float]:
    """Per-layer symmetric color scale, shared by before/after: 98th percentile of |h.d| over the untrusted tokens
    (the text under inspection), or over all shown tokens if nothing is untrusted. Brighter values saturate."""
    idx = torch.tensor(s.untrusted if any(s.untrusted) else s.content, dtype=torch.bool)
    out = []
    for l in range(s.before.shape[0]):
        v = torch.cat([s.before[l, idx], s.after[l, idx]]).abs()
        out.append(float(torch.quantile(v, 0.98)) if len(v) else 1.0)
    return [x if x > 0 else 1.0 for x in out]


def _color(v: float, scale: float) -> str:
    """Diverging red (instruction-like, +) / blue (content-like, -), white at 0."""
    t = max(-1.0, min(1.0, v / scale))
    if t >= 0:
        g = round(255 - 160 * t)
        return f"rgb(255,{g},{g})"
    g = round(255 + 160 * t)
    return f"rgb({g},{g},255)"


_STYLE = """
:root{--bg:#fbfbf8;--fg:#1d1d1b;--muted:#6b6b66;--rule:#dcdcd4}
@media (prefers-color-scheme:dark){:root{--bg:#191917;--fg:#ecece6;--muted:#a3a39b;--rule:#3a3a36}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,system-ui,sans-serif;margin:0;padding:24px 16px}
main{max-width:960px;margin:0 auto}h1{font-size:20px;margin:0 0 4px}p.sub{color:var(--muted);margin:0 0 16px}
.controls{display:flex;flex-wrap:wrap;gap:16px;align-items:center;margin:12px 0 16px;color:var(--muted)}
.panel{font:14px/1.9 ui-monospace,SFMono-Regular,Menlo,monospace;white-space:pre-wrap;word-break:break-word;
border:1px solid var(--rule);border-radius:8px;padding:14px;margin-bottom:12px}
.panel h2{font:600 13px system-ui,sans-serif;margin:0 0 8px;color:var(--muted)}
.t{color:#111;border-radius:3px;padding:1px 0}.u{border-bottom:2px solid #444}
.legend{display:flex;align-items:center;gap:8px;font-size:13px;color:var(--muted)}
.bar{width:160px;height:10px;border-radius:5px;background:linear-gradient(90deg,rgb(95,95,255),#fff,rgb(255,95,95))}
"""

_SCRIPT = """
const D=JSON.parse(document.getElementById('nxdata').textContent);
const lay=document.getElementById('layer'),lab=document.getElementById('layerlabel');
function col(v,s){let t=Math.max(-1,Math.min(1,v/s));if(t>=0){const g=Math.round(255-160*t);return `rgb(255,${g},${g})`}
const g=Math.round(255+160*t);return `rgb(${g},${g},255)`}
function paint(){const l=+lay.value,s=D.scale[l];lab.textContent=`layer ${l}${D.hooked.includes(l)?'':' (not edited)'}`;
for(const which of ['before','after']){document.querySelectorAll(`#${which} .t`).forEach(el=>{const i=+el.dataset.i;
const v=D[which][l][i];el.style.background=col(v,s);el.title=`h·d = ${v.toFixed(2)}`})}}
lay.addEventListener('input',paint);paint();
"""


def _panel(s: TokenScores, which: str, layer: int, scale: float) -> str:
    vals = (s.before if which == "before" else s.after)[layer]
    spans, prev = [], None
    for i, (tok, show) in enumerate(zip(s.tokens, s.content, strict=True)):
        if not show:
            continue
        if prev is not None and s.message[i] != prev:
            spans.append("<br><br>")
        prev = s.message[i]
        text = html.escape(tok).replace("\n", "<br>")
        cls = "t u" if s.untrusted[i] else "t"
        spans.append(f'<span class="{cls}" data-i="{i}" style="background:{_color(float(vals[i]), scale)}">'
                     f"{text}</span>")
    label = "Undefended" if which == "before" else "With NXDirection (untrusted tokens edited)"
    return f'<section class="panel" id="{which}"><h2>{label}</h2>{"".join(spans)}</section>'


def to_html(s: TokenScores, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    scales = _scales(s)
    layer = s.default_layer
    data = {"before": [[round(float(x), 3) for x in row] for row in s.before],
            "after": [[round(float(x), 3) for x in row] for row in s.after],
            "scale": [round(x, 4) for x in scales], "hooked": s.hooked}
    payload = json.dumps(data, allow_nan=False).replace("<", "\\u003c").replace("&", "\\u0026")
    L = s.before.shape[0]
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>nexec heatmap</title>
<style>{_STYLE}</style></head><body><main>
<h1>{html.escape(s.title or "Instruction-likeness per token")}</h1>
<p class="sub">Each token is shaded by its projection on the instruction direction (h·d) at the chosen layer:
red = reads as an instruction, blue = reads as content. Underlined tokens are marked untrusted.</p>
<div class="controls"><label>Layer <input type="range" id="layer" min="0" max="{L - 1}" value="{layer}"></label>
<span id="layerlabel">layer {layer}</span>
<span class="legend">content <span class="bar"></span> instruction</span></div>
{_panel(s, "before", layer, scales[layer])}
{_panel(s, "after", layer, scales[layer])}
<script type="application/json" id="nxdata">{payload}</script>
<script>{_SCRIPT}</script></main></body></html>
"""
    path.write_text(doc, encoding="utf-8")
    return path


def _layout(s: TokenScores, vals: torch.Tensor, width_chars: int) -> list[list[tuple[int, str, float, bool]]]:
    """Wrap shown tokens into rows of (x, text, value, untrusted) on a character grid of ``width_chars`` columns.
    New messages start a new paragraph; newlines break rows; pieces longer than a row are split across rows."""
    rows, row, n, prev = [], [], 0, None
    for i, (tok, show) in enumerate(zip(s.tokens, s.content, strict=True)):
        if not show:
            continue
        if prev is not None and s.message[i] != prev:
            rows += [row, []]
            row, n = [], 0
            tok = tok.lstrip(" ")
        prev = s.message[i]
        for k, line in enumerate(tok.split("\n")):
            if k:
                rows.append(row)
                row, n = [], 0
            while line:
                if n >= width_chars or (n + len(line) > width_chars and row and len(line) <= width_chars):
                    rows.append(row)
                    row, n = [], 0
                piece, line = line[: width_chars - n], line[width_chars - n:]
                row.append((n, piece, float(vals[i]), s.untrusted[i]))
                n += len(piece)
    rows.append(row)
    return rows


def to_png(s: TokenScores, path, *, layer: int | None = None, width_chars: int = 92) -> Path:
    try:  # object-oriented API only: never touch pyplot or the caller's process-wide backend
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        from matplotlib.patches import Rectangle
    except (ImportError, TypeError) as e:
        raise ImportError("to_png needs matplotlib: install the extra with `pip install nexec[viz]`") from e

    layer = s.default_layer if layer is None else layer
    scale = _scales(s)[layer]

    def lines(vals):
        return _layout(s, vals, width_chars)

    panels = [("Undefended", lines(s.before[layer])), ("With NXDirection", lines(s.after[layer]))]
    n_rows = sum(len(r) for _, r in panels) + 2 * len(panels)
    fig_h = 0.9 + 0.26 * n_rows
    fig_w = 0.105 * width_chars + 0.6
    fig = Figure(figsize=(fig_w, fig_h), dpi=200)
    FigureCanvasAgg(fig)
    unit_in = fig_w / (width_chars + 4)        # inches per character cell on the x axis
    mono_pt = unit_in * 72 / 0.64              # DejaVu Sans Mono advance is ~0.602 em; small margin per cell
    fig.patch.set_facecolor("white")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, width_chars + 4)
    ax.set_ylim(n_rows + 2.2, 0)
    ax.axis("off")
    ax.text(1, 0.9, s.title or "Instruction-likeness per token", fontsize=11, weight="bold", va="center")
    ax.text(1, 1.7, f"h·d at layer {layer}: red = reads as instruction, blue = reads as content; "
                    "underline = untrusted", fontsize=7.5, color="#555", va="center")
    y = 2.8
    for label, rows in panels:
        ax.text(1, y, label, fontsize=9, weight="bold", color="#333", va="center")
        y += 1.0
        for row in rows:
            for x, piece, v, unt in row:  # background boxes first, text on top, so no box clips a glyph
                ax.add_patch(Rectangle((1 + x, y - 0.42), len(piece), 0.84, facecolor=_rgb(_color(v, scale)),
                                       edgecolor="none", zorder=1))
                ax.text(1 + x, y, piece, family="DejaVu Sans Mono", fontsize=mono_pt, va="center", ha="left",
                        zorder=2)
                if unt and piece.strip():
                    ax.plot([1 + x, 1 + x + len(piece)], [y + 0.46, y + 0.46], color="#444", lw=0.6, zorder=3)
            y += 1.0
        y += 1.0
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")
    return path


def _rgb(css: str) -> tuple[float, float, float]:
    r, g, b = (int(x) / 255 for x in css[4:-1].split(","))
    return (r, g, b)
