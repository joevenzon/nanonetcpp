#!/usr/bin/env python3
"""
weights_heatmap.py  –  Visualise PyTorch weight tensors as viridis heatmaps.

Usage:
    python weights_heatmap.py <weights_file> [--out output.html]
"""

import argparse
import math
import sys
import json
import html as html_lib
from pathlib import Path

# ── Lazy torch import with a helpful message ──────────────────────────────────
try:
    import torch
except ImportError:
    sys.exit("Error: PyTorch is not installed.  Run:  pip install torch")

try:
    import numpy as np
except ImportError:
    sys.exit("Error: NumPy is not installed.  Run:  pip install numpy")


# ── Viridis colour map (7-stop interpolated) ──────────────────────────────────

def viridis(t: float) -> tuple:
    """Interpolated viridis — returns (R, G, B) in 0-255."""
    stops = [
        [0.267, 0.005, 0.329],
        [0.282, 0.141, 0.458],
        [0.254, 0.265, 0.530],
        [0.164, 0.471, 0.558],
        [0.134, 0.659, 0.455],
        [0.478, 0.821, 0.212],
        [0.993, 0.906, 0.144],
    ]
    t = max(0.0, min(1.0, t))
    idx = t * (len(stops) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(stops) - 1)
    f = idx - lo
    r = stops[lo][0] * (1 - f) + stops[hi][0] * f
    g = stops[lo][1] * (1 - f) + stops[hi][1] * f
    b = stops[lo][2] * (1 - f) + stops[hi][2] * f
    return (round(r * 255), round(g * 255), round(b * 255))


# ── Helper utilities ───────────────────────────────────────────────────────────

def tensor_to_2d(tensor: torch.Tensor) -> np.ndarray:
    """Squeeze / reshape a tensor to a 2-D float32 NumPy array."""
    t = tensor.detach().cpu().float().numpy()
    if t.ndim == 0:
        t = t.reshape(1, 1)
    elif t.ndim == 1:
        t = t.reshape(1, -1)
    elif t.ndim > 2:
        # Flatten all dims except the first into columns
        t = t.reshape(t.shape[0], -1)
    return t


def normalise(arr: np.ndarray):
    """Linear normalise to [0, 1]; handle all-same-value edge case."""
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def array_to_viridis_css(arr2d: np.ndarray) -> str:
    """
    Return a CSS `linear-gradient` string that approximates a 2-D heatmap by
    stacking row-level gradients in a single element via background-image layers.
    This avoids any JS or canvas and keeps the HTML fully self-contained.
    Instead we'll use an SVG data-URI background.
    """
    norm = normalise(arr2d)
    rows, cols = norm.shape

    # Build a tiny SVG where each cell is a coloured rect.
    # Cap cell count for huge layers so the file stays manageable.
    MAX_CELLS = 4096
    if rows * cols > MAX_CELLS:
        scale = math.sqrt(MAX_CELLS / (rows * cols))
        new_rows = max(1, int(rows * scale))
        new_cols = max(1, int(cols * scale))
        # simple nearest-neighbour downsample
        row_idx = (np.linspace(0, rows - 1, new_rows)).astype(int)
        col_idx = (np.linspace(0, cols - 1, new_cols)).astype(int)
        norm = norm[np.ix_(row_idx, col_idx)]
        rows, cols = new_rows, new_cols

    CELL = 16  # px per cell in the SVG
    svg_w = cols * CELL
    svg_h = rows * CELL

    rects = []
    for r in range(rows):
        for c in range(cols):
            v = norm[r, c]
            R, G, B = viridis(v)
            rects.append(
                f'<rect x="{c*CELL}" y="{r*CELL}" width="{CELL}" height="{CELL}" '
                f'fill="rgb({R},{G},{B})"/>'
            )

    svg_body = "".join(rects)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{svg_w}" height="{svg_h}" '
        f'viewBox="0 0 {svg_w} {svg_h}">'
        f'{svg_body}'
        f'</svg>'
    ), rows, cols, svg_w, svg_h


def viridis_legend_svg(width=200, height=18) -> str:
    """Horizontal viridis colour-bar SVG."""
    stops = []
    for i in range(64):  # 64 stops
        t = i / 63
        R, G, B = viridis(t)
        pct = round(i / 63 * 100, 1)
        stops.append(f'<stop offset="{pct}%" stop-color="rgb({R},{G},{B})"/>')
    gradient = f'<linearGradient id="vir">{"".join(stops)}</linearGradient>'
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        f'<defs>{gradient}</defs>'
        f'<rect width="{width}" height="{height}" fill="url(#vir)" rx="3"/>'
        f'</svg>'
    )


# ── HTML template ──────────────────────────────────────────────────────────────

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Weight Heatmaps – {title}</title>
<style>
  :root {{
    --bg: #0f1117;
    --surface: #1a1d27;
    --border: #2e3148;
    --text: #e2e4f0;
    --muted: #8b8fa8;
    --accent: #7c6af7;
    --accent2: #4ecdc4;
  }}

  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    line-height: 1.5;
    padding: 2rem;
  }}

  header {{
    margin-bottom: 2.5rem;
    border-bottom: 1px solid var(--border);
    padding-bottom: 1.5rem;
  }}
  header h1 {{ font-size: 1.6rem; font-weight: 700; color: var(--text); margin-bottom: .25rem; }}
  header p  {{ color: var(--muted); font-size: .9rem; }}

  .summary-grid {{
    display: flex;
    flex-wrap: wrap;
    gap: .75rem;
    margin-bottom: 2.5rem;
  }}
  .stat {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: .65rem 1.1rem;
    min-width: 120px;
  }}
  .stat-label {{ font-size: .72rem; color: var(--muted); text-transform: uppercase; letter-spacing: .06em; }}
  .stat-value {{ font-size: 1.15rem; font-weight: 600; color: var(--accent); }}

  .search-row {{
    display: flex;
    gap: .75rem;
    margin-bottom: 1.5rem;
    flex-wrap: wrap;
    align-items: center;
  }}
  .search-row input {{
    flex: 1;
    min-width: 200px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: .5rem .9rem;
    color: var(--text);
    font-size: .9rem;
    outline: none;
  }}
  .search-row input:focus {{ border-color: var(--accent); }}
  .search-row label {{ color: var(--muted); font-size: .85rem; white-space: nowrap; }}
  .search-row select {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: .5rem .75rem;
    color: var(--text);
    font-size: .85rem;
    outline: none;
  }}
  .search-row select:focus {{ border-color: var(--accent); }}

  .layers {{}}
  .layer-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    margin-bottom: 1.5rem;
    overflow: hidden;
    transition: border-color .2s;
  }}
  .layer-card:hover {{ border-color: var(--accent); }}

  .layer-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: .85rem 1.25rem;
    cursor: pointer;
    user-select: none;
    gap: 1rem;
    flex-wrap: wrap;
  }}
  .layer-name {{
    font-size: .9rem;
    font-weight: 600;
    font-family: 'Cascadia Code', 'Fira Code', 'Consolas', monospace;
    color: var(--accent2);
    word-break: break-all;
  }}
  .layer-meta {{
    display: flex;
    gap: .6rem;
    flex-wrap: wrap;
  }}
  .badge {{
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: .2rem .55rem;
    font-size: .75rem;
    color: var(--muted);
    white-space: nowrap;
  }}
  .badge.shape {{ color: var(--text); }}
  .chevron {{
    font-size: 1rem;
    color: var(--muted);
    transition: transform .25s;
    flex-shrink: 0;
  }}
  .layer-card.open .chevron {{ transform: rotate(180deg); }}

  .layer-body {{
    display: none;
    padding: 1rem 1.25rem 1.25rem;
    border-top: 1px solid var(--border);
  }}
  .layer-card.open .layer-body {{ display: block; }}

  .stats-row {{
    display: flex;
    flex-wrap: wrap;
    gap: .6rem;
    margin-bottom: 1rem;
  }}
  .mini-stat {{
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 7px;
    padding: .3rem .65rem;
    font-size: .78rem;
  }}
  .mini-stat span {{ color: var(--muted); }}
  .mini-stat strong {{ color: var(--text); }}

  .heatmap-wrap {{
    overflow: auto;
    border-radius: 8px;
    background: var(--bg);
    padding: .5rem;
    max-height: 520px;
  }}
  .heatmap-wrap svg {{ display: block; image-rendering: pixelated; }}

  .legend-row {{
    display: flex;
    align-items: center;
    gap: .75rem;
    margin-top: .75rem;
    font-size: .75rem;
    color: var(--muted);
  }}
  .legend-row svg {{ flex-shrink: 0; }}

  #no-match {{
    display: none;
    color: var(--muted);
    padding: 2rem;
    text-align: center;
    font-size: .95rem;
  }}
</style>
</head>
<body>

<header>
  <h1>⚡ Weight Heatmaps</h1>
  <p>Source file: <code>{filename}</code></p>
</header>

<div class="summary-grid">
  <div class="stat"><div class="stat-label">Layers</div><div class="stat-value">{n_layers}</div></div>
  <div class="stat"><div class="stat-label">Total Params</div><div class="stat-value">{total_params}</div></div>
  <div class="stat"><div class="stat-label">File Size</div><div class="stat-value">{file_size}</div></div>
</div>

<div class="search-row">
  <input id="search" type="text" placeholder="Filter layers by name…" oninput="filterLayers()"/>
  <label>Sort by</label>
  <select id="sort-sel" onchange="sortLayers()">
    <option value="original">Original order</option>
    <option value="name">Name A→Z</option>
    <option value="params-desc">Params (most first)</option>
    <option value="params-asc">Params (fewest first)</option>
  </select>
</div>

<div class="layers" id="layers">
{layer_cards}
</div>
<div id="no-match">No layers match your filter.</div>

<script>
const allCards = Array.from(document.querySelectorAll('.layer-card'));

function filterLayers() {{
  const q = document.getElementById('search').value.toLowerCase();
  let shown = 0;
  allCards.forEach(c => {{
    const name = c.dataset.name;
    const vis = !q || name.includes(q);
    c.style.display = vis ? '' : 'none';
    if (vis) shown++;
  }});
  document.getElementById('no-match').style.display = shown ? 'none' : 'block';
}}

function sortLayers() {{
  const mode = document.getElementById('sort-sel').value;
  const container = document.getElementById('layers');
  const cards = [...allCards];
  if (mode === 'name') {{
    cards.sort((a,b) => a.dataset.name.localeCompare(b.dataset.name));
  }} else if (mode === 'params-desc') {{
    cards.sort((a,b) => +b.dataset.params - +a.dataset.params);
  }} else if (mode === 'params-asc') {{
    cards.sort((a,b) => +a.dataset.params - +b.dataset.params);
  }} else {{
    cards.sort((a,b) => +a.dataset.idx - +b.dataset.idx);
  }}
  cards.forEach(c => container.appendChild(c));
  filterLayers();
}}

function toggle(id) {{
  const card = document.getElementById(id);
  card.classList.toggle('open');
}}
</script>
</body>
</html>
"""

CARD_TEMPLATE = """\
<div class="layer-card open" id="{card_id}" data-name="{data_name}" data-params="{n_params}" data-idx="{idx}">
  <div class="layer-header" onclick="toggle('{card_id}')">
    <span class="layer-name">{layer_name}</span>
    <div class="layer-meta">
      <span class="badge shape">{shape}</span>
      <span class="badge">{dtype}</span>
      <span class="badge">{n_params} params</span>
    </div>
    <span class="chevron">▾</span>
  </div>
  <div class="layer-body">
    <div class="stats-row">
      <div class="mini-stat"><span>min </span><strong>{vmin:.6g}</strong></div>
      <div class="mini-stat"><span>max </span><strong>{vmax:.6g}</strong></div>
      <div class="mini-stat"><span>mean </span><strong>{vmean:.6g}</strong></div>
      <div class="mini-stat"><span>std </span><strong>{vstd:.6g}</strong></div>
      <div class="mini-stat"><span>displayed </span><strong>{disp_rows}×{disp_cols} cells</strong></div>
    </div>
    <div class="heatmap-wrap">
{svg_content}
    </div>
    <div class="legend-row">
      {legend_svg}
      <span>{vmin:.4g}</span>
      <span style="flex:1;text-align:center">viridis — low → high</span>
      <span>{vmax:.4g}</span>
    </div>
  </div>
</div>
"""


# ── Main ───────────────────────────────────────────────────────────────────────

def fmt_num(n: int) -> str:
    """Format large numbers with K/M/B suffix."""
    for div, suf in [(1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K")]:
        if n >= div:
            return f"{n/div:.2f}{suf}"
    return str(n)


def fmt_bytes(b: int) -> str:
    for div, suf in [(1 << 30, "GB"), (1 << 20, "MB"), (1 << 10, "KB")]:
        if b >= div:
            return f"{b/div:.1f} {suf}"
    return f"{b} B"


def load_state_dict(path: str):
    """Load a state-dict or raw tensor dict from a .pt/.pth file."""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    # If it's a plain tensor, wrap it.
    if isinstance(obj, torch.Tensor):
        return {"weight": obj}
    # If it has a 'state_dict' attribute/key (e.g. a checkpoint dict), unwrap it.
    if isinstance(obj, dict):
        if "state_dict" in obj:
            return obj["state_dict"]
        if "model_state_dict" in obj:
            return obj["model_state_dict"]
        # Filter to only tensor values
        tensors = {k: v for k, v in obj.items() if isinstance(v, torch.Tensor)}
        if tensors:
            return tensors
    # Try calling .state_dict() (a full nn.Module)
    if hasattr(obj, "state_dict"):
        return obj.state_dict()
    raise ValueError(
        f"Could not extract a weight dict from the loaded object (type: {type(obj).__name__})."
    )


def build_html(weights_path: str, out_path: str):
    path = Path(weights_path)
    if not path.exists():
        sys.exit(f"Error: file not found: {weights_path}")

    print(f"Loading weights from: {path}")
    state_dict = load_state_dict(str(path))

    n_layers = len(state_dict)
    total_params = sum(v.numel() for v in state_dict.values())
    file_size = fmt_bytes(path.stat().st_size)
    legend_svg = viridis_legend_svg(180, 16)

    print(f"Found {n_layers} tensors  ({fmt_num(total_params)} parameters)")

    cards_html = []
    for idx, (name, tensor) in enumerate(state_dict.items()):
        arr2d = tensor_to_2d(tensor)
        orig_rows, orig_cols = arr2d.shape
        n_params = tensor.numel()
        vmin, vmax = float(arr2d.min()), float(arr2d.max())
        vmean = float(arr2d.mean())
        vstd = float(arr2d.std())

        svg_content, disp_rows, disp_cols, svg_w, svg_h = array_to_viridis_css(arr2d)

        shape_str = "×".join(str(d) for d in tensor.shape) if tensor.ndim > 0 else "scalar"
        dtype_str = str(tensor.dtype).replace("torch.", "")
        card_id = f"layer_{idx}"

        cards_html.append(CARD_TEMPLATE.format(
            card_id=card_id,
            data_name=html_lib.escape(name.lower()),
            n_params=n_params,
            idx=idx,
            layer_name=html_lib.escape(name),
            shape=shape_str,
            dtype=dtype_str,
            vmin=vmin,
            vmax=vmax,
            vmean=vmean,
            vstd=vstd,
            disp_rows=disp_rows,
            disp_cols=disp_cols,
            svg_content=svg_content,
            legend_svg=legend_svg,
        ))

        print(f"  [{idx+1:>4}/{n_layers}]  {name:60s}  {shape_str}")

    html = HTML_TEMPLATE.format(
        title=html_lib.escape(path.name),
        filename=html_lib.escape(str(path)),
        n_layers=n_layers,
        total_params=fmt_num(total_params),
        file_size=file_size,
        layer_cards="\n".join(cards_html),
    )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"\n✓  Heatmap HTML written to: {out}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualise PyTorch model weights as viridis heatmaps in an HTML file."
    )
    parser.add_argument("weights", help="Path to the .pt / .pth weights file (torch.save output)")
    parser.add_argument(
        "--out", default="python_weights_heatmap.html",
        help="Output HTML file path (default: python_weights_heatmap.html)"
    )
    args = parser.parse_args()
    build_html(args.weights, args.out)


if __name__ == "__main__":
    main()
