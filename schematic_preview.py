"""Compact vector well schematic used by the main T&D Streamlit application."""
from html import escape


def build_well_schematic_svg(
    td=2500.0, section_count=4, casing_shoe=550.0,
    liner1_top=500.0, liner1_shoe=1300.0,
    liner2_top=1250.0, liner2_shoe=2100.0,
    kop=1600.0, eob=2000.0, final_inc=30.0,
    operation="Drilling", dp_length=1600.0, hwdp_length=400.0,
    dc_length=500.0, running_tubular_length=0.0,
    casing_label='20 in × 94 ppf | ID 18.730 in',
    liner1_label='13-3/8 in × 72 ppf | ID 12.347 in',
    liner2_label='9-5/8 in × 53.5 ppf | ID 8.835 in',
    open_hole_label='Open hole | 8.75 in',
):
    td = max(float(td), 100.0)
    section_count = int(section_count)
    casing_shoe = min(max(float(casing_shoe), 1.0), td)
    liner1_top = min(max(float(liner1_top), 0.0), td)
    liner1_shoe = min(max(float(liner1_shoe), liner1_top + 1.0), td)
    liner2_top = min(max(float(liner2_top), 0.0), td)
    liner2_shoe = min(max(float(liner2_shoe), liner2_top + 1.0), td)
    kop = min(max(float(kop), 0.0), td)
    eob = min(max(float(eob), kop), td)
    final_inc = max(0.0, min(float(final_inc), 90.0))

    W, H = 760, 900
    top, bottom = 62, 805
    plot_h = bottom - top
    x_ruler, x_center = 62, 235
    y = lambda d: top + (min(max(float(d), 0.0), td) / td) * plot_h

    open_top = casing_shoe
    if section_count >= 3:
        open_top = max(open_top, liner1_shoe)
    if section_count >= 4:
        open_top = max(open_top, liner2_shoe)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" height="100%" preserveAspectRatio="xMidYMin meet" role="img" aria-label="Interactive well schematic">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#202124}.small{font-size:11px}.label{font-size:12px;font-weight:700}.muted{fill:#62676d}.marker{font-size:12px;font-weight:700}</style>',
        '<text x="10" y="24" font-size="14" font-weight="700">Depth (mMD)</text>',
        f'<text x="{x_center}" y="24" text-anchor="middle" font-size="14" font-weight="700">Well Schematic</text>',
        '<text x="605" y="24" font-size="13" font-weight="700">String / Markers</text>',
    ]

    # Depth ruler.
    parts.append(f'<line x1="{x_ruler}" y1="{top}" x2="{x_ruler}" y2="{bottom}" stroke="#111" stroke-width="1.8"/>')
    key_depths = [casing_shoe, td, kop, eob]
    if section_count >= 3: key_depths += [liner1_top, liner1_shoe]
    if section_count >= 4: key_depths += [liner2_top, liner2_shoe]
    ticks = set(range(0, int(td)+1, 100))
    ticks.update(int(round(v)) for v in key_depths if 0 <= v <= td)
    for d in sorted(ticks):
        yy = y(d)
        is_key = any(abs(d-v) < .51 for v in key_depths)
        major = d % 500 == 0 or is_key
        x2 = x_ruler + (10 if major else 5)
        parts.append(f'<line x1="{x_ruler-1}" y1="{yy:.1f}" x2="{x2}" y2="{yy:.1f}" stroke="#111" stroke-width="{1.3 if major else .7}"/>')
        if major:
            parts.append(f'<text x="{x_ruler-8}" y="{yy+3.5:.1f}" text-anchor="end" class="small">{d:,}</text>')

    # Walls: casing solid; liner intervals dashed; open hole wavy.
    sections = [(0.0, casing_shoe, 'Cased Section', casing_label, 70, 'cased')]
    if section_count >= 3:
        sections.append((liner1_top, liner1_shoe, 'Liner 1 Section', f'{liner1_label} | TOL {liner1_top:,.0f} m | Shoe {liner1_shoe:,.0f} m', 54, 'liner'))
    if section_count >= 4:
        sections.append((liner2_top, liner2_shoe, 'Liner 2 Section', f'{liner2_label} | TOL {liner2_top:,.0f} m | Shoe {liner2_shoe:,.0f} m', 40, 'liner'))
    sections.append((open_top, td, 'Open Hole Section', open_hole_label, 27, 'open'))

    # Draw tubular walls and section labels.
    for i, (start, end, title, desc, width, kind) in enumerate(sections):
        ys, ye = y(start), y(end)
        left, right = x_center-width/2, x_center+width/2
        if kind == 'open':
            step = 17.0
            count = max(1, int((ye-ys)/step))
            lcmd, rcmd = [f'M {left:.1f} {ys:.1f}'], [f'M {right:.1f} {ys:.1f}']
            for j in range(count):
                y0, y1 = ys+(ye-ys)*j/count, ys+(ye-ys)*(j+1)/count
                ym = (y0+y1)/2
                lcmd.append(f'Q {left-5:.1f} {ym:.1f} {left:.1f} {y1:.1f}')
                rcmd.append(f'Q {right+5:.1f} {ym:.1f} {right:.1f} {y1:.1f}')
            parts.append(f'<path d="{" ".join(lcmd)}" fill="none" stroke="#222" stroke-width="2.2"/>')
            parts.append(f'<path d="{" ".join(rcmd)}" fill="none" stroke="#222" stroke-width="2.2"/>')
        else:
            dash = ' stroke-dasharray="7 5"' if kind == 'liner' else ''
            sw = 2.6 if kind == 'cased' else 2.2
            parts.append(f'<line x1="{left:.1f}" y1="{ys:.1f}" x2="{left:.1f}" y2="{ye:.1f}" stroke="#111" stroke-width="{sw}"{dash}/>')
            parts.append(f'<line x1="{right:.1f}" y1="{ys:.1f}" x2="{right:.1f}" y2="{ye:.1f}" stroke="#111" stroke-width="{sw}"{dash}/>')

        # Filled triangular shoe markers at bottom of casing/liner.
        if kind in ('cased', 'liner'):
            yy = ye
            parts.append(f'<polygon points="{left:.1f},{yy:.1f} {left-7:.1f},{yy+7:.0f} {left:.1f},{yy+7:.0f}" fill="#111111" stroke="#111111" stroke-width="0.8"/>')
            parts.append(f'<polygon points="{right:.1f},{yy:.1f} {right+7:.1f},{yy+7:.0f} {right:.1f},{yy+7:.0f}" fill="#111111" stroke="#111111" stroke-width="0.8"/>')
            # Short shoe ledge for visual clarity.
            parts.append(f'<line x1="{left-1:.1f}" y1="{yy:.1f}" x2="{left+6:.1f}" y2="{yy:.1f}" stroke="#111" stroke-width="1.4"/>')
            parts.append(f'<line x1="{right-6:.1f}" y1="{yy:.1f}" x2="{right+1:.1f}" y2="{yy:.1f}" stroke="#111" stroke-width="1.4"/>')

        # Section guide line to label column.
        parts.append(f'<line x1="{x_center+width/2+8:.1f}" y1="{ye:.1f}" x2="315" y2="{ye:.1f}" stroke="#b7bbc0" stroke-width="0.8" stroke-dasharray="3 4"/>')
        label_x = 325
        label_y = (ys+ye)/2
        label_y = min(max(label_y, ys+20), ye-20) if ye-ys > 40 else (ys+ye)/2
        parts.append(f'<text x="{label_x}" y="{label_y-8:.1f}" class="label">{escape(title)}<title>{escape(title + ' — ' + desc)}</title></text>')
        # Keep label text concise and explicitly list TOL and shoe for liners.
        if kind == 'liner':
            spec = escape(desc.split(' | TOL ')[0])
            parts.append(f'<text x="{label_x}" y="{label_y+8:.1f}" class="small muted">{spec}</text>')
            parts.append(f'<text x="{label_x}" y="{label_y+22:.1f}" class="small muted">TOL {start:,.0f} m • Shoe {end:,.0f} m</text>')
        else:
            parts.append(f'<text x="{label_x}" y="{label_y+8:.1f}" class="small muted">{escape(desc)}</text>')
            parts.append(f'<text x="{label_x}" y="{label_y+22:.1f}" class="small muted">{start:,.0f}–{end:,.0f} mMD</text>')

    # Surface and bottom termination.
    parts.append(f'<line x1="{x_center-38}" y1="{top}" x2="{x_center+38}" y2="{top}" stroke="#111" stroke-width="3"/>')
    parts.append(f'<path d="M {x_center-10} {bottom-2} Q {x_center} {bottom+9} {x_center+10} {bottom-2}" fill="none" stroke="#111" stroke-width="2.5"/>')

    # KOP and EOB annotations beside the ruler.
    for name, depth, note in [('KOP', kop, 'Inc: 0°'), ('EOB', eob, f'Inc: {final_inc:g}°')]:
        yy = y(depth)
        parts.append(f'<line x1="{x_ruler+11}" y1="{yy:.1f}" x2="{x_center-48}" y2="{yy:.1f}" stroke="#dc2626" stroke-width="1.3" stroke-dasharray="5 4"/>')
        parts.append(f'<circle cx="{x_ruler}" cy="{yy:.1f}" r="3.5" fill="#dc2626"/>')
        text_y = yy-9 if yy < bottom-34 else yy-25
        parts.append(f'<rect x="{x_ruler+14}" y="{text_y-11:.1f}" width="108" height="30" fill="#fff" opacity="0.94"/>')
        parts.append(f'<text x="{x_ruler+18}" y="{text_y:.1f}" class="marker" style="fill:#dc2626">{name} @ {depth:,.0f} m</text>')
        parts.append(f'<text x="{x_ruler+18}" y="{text_y+13:.1f}" class="small" style="fill:#dc2626">{note}</text>')

    # String intervals through TD.
    op = operation.lower()
    if op == 'drilling':
        hw = max(0.0, min(float(hwdp_length), td))
        dc = max(0.0, min(float(dc_length), td-hw))
        dp = max(0.0, td-hw-dc)
        comps = [('DP', 0.0, dp, '#2563eb'), ('HWDP', dp, dp+hw, '#f59e0b'), ('DC', dp+hw, td, '#16a34a')]
    elif op == 'running liner':
        liner = max(0.0, min(float(running_tubular_length), td))
        dp = td-liner
        comps = [('DP', 0.0, dp, '#2563eb'), ('Liner', dp, td, '#f59e0b')]
    else:
        comps = [('Casing', 0.0, td, '#2563eb')]
    bx = 590
    for name, start, end, color in comps:
        if end <= start: continue
        parts.append(f'<line x1="{x_center}" y1="{y(start):.1f}" x2="{x_center}" y2="{y(end):.1f}" stroke="{color}" stroke-width="4" stroke-linecap="round"><title>{escape(name)}: {start:,.0f}–{end:,.0f} mMD</title></line>')
        y1, y2 = y(start), y(end)
        parts.append(f'<line x1="{bx}" y1="{y1:.1f}" x2="{bx}" y2="{y2:.1f}" stroke="{color}" stroke-width="2.2"/>')
        parts.append(f'<line x1="{bx-5}" y1="{y1:.1f}" x2="{bx+5}" y2="{y1:.1f}" stroke="{color}" stroke-width="1.8"/>')
        parts.append(f'<line x1="{bx-5}" y1="{y2:.1f}" x2="{bx+5}" y2="{y2:.1f}" stroke="{color}" stroke-width="1.8"/>')
        ty = min(max((y1+y2)/2-3, 48), H-50)
        parts.append(f'<text x="605" y="{ty:.1f}" class="label" style="fill:{color}">{escape(name)}</text>')
        parts.append(f'<text x="605" y="{ty+15:.1f}" class="small">{start:,.0f}–{end:,.0f} m</text>')
        parts.append(f'<text x="605" y="{ty+29:.1f}" class="small">Length: {end-start:,.0f} m</text>')

    parts.append(f'<text x="{x_center+17}" y="{bottom+2}" class="marker">TD @ {td:,.0f} m</text>')
    parts.append('<text x="10" y="878" class="small muted">Schematic not to scale horizontally • Depth positions are proportional to MD</text>')
    parts.append('</svg>')
    return ''.join(parts)
