"""Draws the per-step training curve of a run as a static SVG for the handout.
    python3 docs/handout/plot_steps.py docs/results/qwen3b_mt_leg1_steps.csv out.svg
Three proportions on one 0-1 axis; palette = dataviz reference categorical
slots 1-3 (validated: CVD dE 9.2, aqua below 3:1 so every line is directly
labeled and the handout carries the numbers as a table).
"""

from __future__ import annotations

import csv
import sys

SERIES = [  # (column, label, color)
    ("format_ok", "按格式交卷的比例", "#2a78d6"),
    ("train_acc", "训练题答对的比例", "#eb6834"),
    ("signal_share", "有学习信号的组的比例", "#1baf7a"),
]
W, H = 940, 420
L, R, T, B = 64, 190, 24, 56  # plot margins; right margin holds the direct labels
INK, MUTED, GRID, SURFACE = "#1f2328", "#59636e", "#e6e8eb", "#fcfcfb"


def main() -> None:
    rows = [r for r in csv.DictReader(line for line in open(sys.argv[1]) if not line.startswith("#"))]
    steps = [int(r["step"]) for r in rows]
    x0, x1 = 0, max(steps)

    def x(s: float) -> float:
        return L + (s - x0) / (x1 - x0) * (W - L - R)

    def y(v: float) -> float:
        return T + (1 - v) * (H - T - B)

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
           f'font-family="-apple-system,BlinkMacSystemFont,\'PingFang SC\',\'Microsoft YaHei\',sans-serif">',
           f'<rect width="{W}" height="{H}" fill="{SURFACE}"/>']
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        out.append(f'<line x1="{L}" x2="{W - R}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="{GRID}" stroke-width="1"/>')
        out.append(f'<text x="{L - 8}" y="{y(v) + 4:.1f}" text-anchor="end" font-size="12" fill="{MUTED}">{v:.0%}</text>')
    for s in range(0, x1 + 1, 5):
        out.append(f'<text x="{x(s):.1f}" y="{H - B + 20}" text-anchor="middle" font-size="12" fill="{MUTED}">{s}</text>')
    out.append(f'<text x="{(L + W - R) / 2:.1f}" y="{H - 12}" text-anchor="middle" font-size="13" fill="{MUTED}">训练步数</text>')
    # the phase boundary the handout discusses
    out.append(f'<line x1="{x(9):.1f}" x2="{x(9):.1f}" y1="{T}" y2="{H - B}" stroke="{MUTED}" stroke-width="1" stroke-dasharray="3 3"/>')
    out.append(f'<text x="{x(9) + 6:.1f}" y="{y(0.62):.1f}" font-size="12" fill="{MUTED}">第 9 步：格式学会了</text>')

    ends = []
    for col, label, color in SERIES:
        pts = " ".join(f"{x(s):.1f},{y(float(r[col])):.1f}" for s, r in zip(steps, rows, strict=True))
        out.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round"/>')
        last = float(rows[-1][col])
        out.append(f'<circle cx="{x(steps[-1]):.1f}" cy="{y(last):.1f}" r="4" fill="{color}" stroke="{SURFACE}" stroke-width="2"/>')
        ends.append([y(last), label, color, last])
    # direct labels at the right end, nudged apart so they never overlap
    ends.sort()
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 18)
    for ly, label, color, last in ends:
        out.append(f'<rect x="{W - R + 12}" y="{ly - 5:.1f}" width="10" height="10" rx="2" fill="{color}"/>')
        out.append(f'<text x="{W - R + 28}" y="{ly + 4:.1f}" font-size="12" fill="{INK}">{label} {last:.0%}</text>')
    out.append("</svg>")
    open(sys.argv[2], "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
