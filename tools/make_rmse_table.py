#!/usr/bin/env python
"""Regenerate the RMSE Summary tables in tpxo.html.

The two 2D barotropic reference columns are transcribed constants -- they come
from the four-panel figures, which were produced elsewhere. The 3D baroclinic
columns are read from tpxo_metrics.json, written by make_tpxo_plots.py.

Rewrites everything between the RMSE-TABLES:BEGIN/END markers, so the three
region panels stay consistent with the maps instead of being hand-edited.

    python tools/make_rmse_table.py

Needs no special environment; plain python3.
"""

import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
HTML = os.path.join(REPO, "tpxo.html")
METRICS = os.path.join(HERE, "tpxo_metrics.json")

BEGIN = "<!-- RMSE-TABLES:BEGIN"
END = "<!-- RMSE-TABLES:END -->"

CONSTITUENTS = ["M2", "S2", "N2", "K1", "O1"]
REGIONS = [
    ("global", "region-global", "Global", True),
    ("deep", "region-deep", "Deep", False),
    ("shallow", "region-shallow", "Shallow", False),
]

# 2D barotropic reference runs: full-year, complex RMSE in cm, as printed on
# their four-panel figures. Not recomputed here.
TWOD = {
    "bpanomaly": {
        "global": [8.142, 3.582, 1.654, 2.496, 1.858],
        "deep": [7.135, 3.065, 1.426, 1.261, 1.112],
        "shallow": [15.373, 6.825, 3.292, 6.626, 4.530],
    },
    "atm-tide-only": {
        "global": [4.979, 2.117, 0.920, 1.861, 1.336],
        "deep": [2.695, 1.336, 0.542, 1.134, 0.686],
        "shallow": [14.176, 4.553, 2.475, 3.973, 2.946],
    },
}

# (key, header, sub-label, badge colour, group). Group 2d/3d controls both the
# divider and which cells compete for the best/worst highlight -- the 3D runs
# are 45-day baroclinic and are not like-for-like with the 2D year-long refs.
COLUMNS = [
    ("bpanomaly", "BP Anomaly", "1yr 2D", "#2ca02c", "2d"),
    ("atm-tide-only", "Atm Only", "1yr 2D", "#1f77b4", "2d"),
    ("betaSAL-45d", "&beta; SAL", "45d 3D", "#d62728", "3d"),
    ("SAL29-45d", "SAL n=29", "45d 3D", "#d62728", "3d"),
    ("SAL29-loglaw-45d", "+ log-law", "45d 3D", "#d62728", "3d"),
    ("SAL29-loglaw-dt600-45d", "+ &Delta;t=600", "45d 3D", "#9467bd", "3d"),
    ("SAL29-loglaw-powerlaw-45d", "+ S&amp;McW", "45d 3D", "#17becf", "3d"),
    ("JSL-drag-66d", "+ JSL drag", "45d 3D", "#e377c2", "3d"),
    ("JSL-bandpass-66d", "+ bandpass", "45d 3D", "#e377c2", "3d"),
]


def value(metrics, key, region, ci):
    if key in TWOD:
        return TWOD[key][region][ci]
    return metrics[key][CONSTITUENTS[ci]][region]["rmse_cm"]


def build_panel(metrics, region, panel_id, active):
    head = [
        '          <tr>',
        '            <th>Con</th>',
    ]
    for key, label, sub, colour, group in COLUMNS:
        cls = ' class="grp"' if key == "betaSAL-45d" else ""
        head.append(
            f'            <th{cls}><span class="run-badge" style="background:{colour}">'
            f"</span>{label}<br><small>{sub}</small></th>"
        )
    head.append("          </tr>")

    body = []
    for ci, con in enumerate(CONSTITUENTS):
        row = [f"          <tr><td>{con}</td>"]
        vals = {k: value(metrics, k, region, ci) for k, _, _, _, _ in COLUMNS}
        marks = {}
        for group in ("2d", "3d"):
            members = [k for k, _, _, _, g in COLUMNS if g == group]
            lo = min(members, key=lambda k: vals[k])
            hi = max(members, key=lambda k: vals[k])
            marks[lo] = "best"
            marks[hi] = "worst"
        for key, _, _, _, _ in COLUMNS:
            classes = [c for c in (marks.get(key), "grp" if key == "betaSAL-45d" else None) if c]
            attr = f' class="{" ".join(classes)}"' if classes else ""
            row.append(f"<td{attr}>{vals[key]:.3f}</td>")
        row.append("</tr>")
        body.append(" ".join(row))

    cls = "region-panel active" if active else "region-panel"
    return "\n".join(
        [
            f'    <div class="{cls}" id="{panel_id}">',
            '      <div class="table-wrap">',
            '        <table class="rmse-table">',
            "          <thead>",
            *head,
            "          </thead>",
            "          <tbody>",
            *body,
            "          </tbody>",
            "        </table>",
            "      </div>",
            "    </div>",
        ]
    )


def main():
    with open(METRICS) as fh:
        metrics = json.load(fh)

    panels = []
    for region, panel_id, comment, active in REGIONS:
        panels.append(f"    <!-- {comment} -->")
        panels.append(build_panel(metrics, region, panel_id, active))
    block = "\n".join(panels)

    with open(HTML, encoding="utf-8") as fh:
        html = fh.read()

    start = html.index(BEGIN)
    start = html.index("-->", start) + len("-->")
    end = html.index(END)
    new = html[:start] + "\n" + block + "\n    " + html[end:]

    with open(HTML, "w", encoding="utf-8") as fh:
        fh.write(new)

    print(f"regenerated {len(REGIONS)} panels x {len(COLUMNS)} columns")
    for region, _, comment, _ in REGIONS:
        best = min(
            (k for k, _, _, _, g in COLUMNS if g == "3d"),
            key=lambda k: metrics[k]["M2"][region]["rmse_cm"],
        )
        print(f"  {comment:8s} best 3D on M2: {best} "
              f"({metrics[best]['M2'][region]['rmse_cm']:.3f} cm)")


if __name__ == "__main__":
    main()
