from io import BytesIO
from datetime import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image, KeepTogether
)


def _fmt(x, nd=2):
    if x is None:
        return '—'
    if isinstance(x, str):
        return x
    return f'{float(x):,.{nd}f}'


def _page_header_footer(logo_path=None, report_date=None, project_name=''):
    """Header/footer styled after the supplied Rigsis report example."""
    def draw(canvas, doc):
        canvas.saveState()
        # Header: report title at left, Rigsis logo at right, thin rule below.
        canvas.setFont('Helvetica-Bold', 8)
        canvas.setFillColor(colors.HexColor('#222222'))
        canvas.drawString(15*mm, 285.5*mm, 'Preliminary Torque & Drag Analysis')
        if logo_path and os.path.exists(logo_path):
            try:
                canvas.drawImage(
                    logo_path, 181*mm, 282.2*mm,
                    width=13.44*mm, height=8.16*mm,
                    preserveAspectRatio=True, mask='auto'
                )
            except Exception:
                pass
        canvas.setStrokeColor(colors.HexColor('#CFCFCF'))
        canvas.setLineWidth(0.45)
        canvas.line(15*mm, 280.8*mm, 195*mm, 280.8*mm)

        # Footer: report date left, project center, page right.
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(colors.HexColor('#666666'))
        canvas.drawString(15*mm, 9.5*mm, report_date or '')
        canvas.drawCentredString(105*mm, 9.5*mm, project_name or '')
        canvas.drawRightString(195*mm, 9.5*mm, f'Page {doc.page}')
        canvas.restoreState()
    return draw


def _cover_page(story, report_info, inputs, icon_path, logo_path, styles):
    story.append(Spacer(1, 7*mm))
    if logo_path and os.path.exists(logo_path):
        story.append(Image(logo_path, width=48*mm, height=28*mm, hAlign='CENTER'))
    story.append(Spacer(1, 8*mm))
    story.append(Paragraph('<b>PRELIMINARY</b>', styles['CoverSmall']))
    story.append(Paragraph('TORQUE &amp; DRAG ANALYSIS', styles['CoverTitle']))
    story.append(Spacer(1, 7*mm))
    if icon_path and os.path.exists(icon_path):
        story.append(Image(icon_path, width=24*mm, height=24*mm, hAlign='CENTER'))
        story.append(Spacer(1, 4*mm))
    story.append(Paragraph(f"Well: {report_info['well_name']}", styles['CoverMeta']))
    story.append(Paragraph(f"Project / Field: {report_info['project_name']}", styles['CoverMeta']))
    story.append(Paragraph(f"Location: {report_info['location']}", styles['CoverMeta']))
    story.append(Spacer(1, 30*mm))
    story.append(Paragraph(f"Prepared by: {report_info['prepared_by']}", styles['CoverMeta']))
    story.append(Paragraph(f"Report Date: {datetime.now().strftime('%d %B %Y')}", styles['CoverMeta']))
    story.append(Paragraph(f"Report No.: {report_info['report_number']} | Revision: 0", styles['CoverMeta']))
    story.append(PageBreak())


def _summary_table(report_info, inputs, result_df):
    last = result_df.iloc[-1]
    rows = [
        ['Parameter', 'Value'],
        ['Project / Field Name', report_info['project_name']],
        ['Client / Company', report_info['client']],
        ['Location', report_info['location']],
        ['Prepared By', report_info['prepared_by']],
        ['Report Number', str(report_info['report_number'])],
        ['Assessment Type', report_info['assessment_type']],
        ['Operation', inputs.get('analysis_string', 'Drilling')],
        ['Well Name', report_info['well_name']],
        ['Total Depth', f"{_fmt(inputs['td'], 0)} m"],
        ['KOP / EOB', f"{_fmt(inputs['kop'], 0)} m / {_fmt(inputs['eob'], 0)} m"],
        ['Final Inclination', f"{_fmt(inputs['final_inc'], 2)}°"],
        ['Previous Casing', inputs['casing_spec']],
        ['Previous Casing Shoe', f"{_fmt(inputs['casing_shoe'], 0)} mMD"],
        ['Well Section Configuration', f"{inputs.get('section_count', 2)} sections"],
        ['Open Hole ID', f"{_fmt(inputs['open_hole_id'], 3)} in"],
        ['Mud Volume in Hole', f"{_fmt(inputs.get('mud_fill_percent', 100.0), 0)}%"],
        ['Nominal Casing FF', _fmt(inputs['ff_casing_nom'], 2)],
        ['Nominal Open Hole FF', _fmt(inputs['ff_open_nom'], 2)],
        ['RPM', f"{_fmt(inputs['rpm'], 0)} rpm"],
        ['WOB', f"{_fmt(inputs['wob'], 0)} lbf"],
        ['Bit Torque', f"{_fmt(inputs['bit_torque'], 0)} lbf-ft"],
        ['PUW @ Last MD', f"{_fmt(last['PUW_lbf'], 0)} lbf"],
        ['ROTW @ Last MD', f"{_fmt(last['ROTW_lbf'], 0)} lbf"],
        ['SOW @ Last MD', f"{_fmt(last['SOW_lbf'], 0)} lbf"],
        ['Surface Torque @ Last MD', f"{_fmt(last['SurfaceTorque_lbf_ft'], 0)} lbf-ft"],
    ]
    return rows


def _input_table(inputs):
    rows = [
        ['Group', 'Parameter', 'Value', 'Unit'],
        ['Well Geometry', 'TD', _fmt(inputs['td'], 2), 'm'],
        ['Well Geometry', 'KOP', _fmt(inputs['kop'], 2), 'm'],
        ['Well Geometry', 'EOB', _fmt(inputs['eob'], 2), 'm'],
        ['Well Geometry', 'Final Inclination', _fmt(inputs['final_inc'], 2), '°'],
        ['Well Geometry', 'Previous Casing Shoe', _fmt(inputs['casing_shoe'], 2), 'mMD'],
        ['Well Geometry', 'Previous Casing', inputs['casing_spec'], 'API selection'],
        ['Well Geometry', 'Previous Casing ID', _fmt(inputs['casing_id'], 3), 'in'],
        ['Well Geometry', 'Open Hole ID', _fmt(inputs['open_hole_id'], 3), 'in'],
        ['Well Geometry', 'Number of Sections', str(inputs.get('section_count', 2)), '—'],
        *([[ 'Well Geometry', 'Liner 1 Specification', inputs.get('liner1_spec', '—'), 'API selection'], ['Well Geometry', 'Liner 1 TOL', _fmt(inputs.get('liner1_top', 0), 2), 'mMD'], ['Well Geometry', 'Liner 1 Shoe Depth', _fmt(inputs.get('liner1_shoe', 0), 2), 'mMD'], ['Well Geometry', 'Liner 1 ID', _fmt(inputs.get('liner1_id', 0), 3), 'in']] if int(inputs.get('section_count', 2)) >= 3 else []),
        *([[ 'Well Geometry', 'Liner 2 Specification', inputs.get('liner2_spec', '—'), 'API selection'], ['Well Geometry', 'Liner 2 TOL', _fmt(inputs.get('liner2_top', 0), 2), 'mMD'], ['Well Geometry', 'Liner 2 Shoe Depth', _fmt(inputs.get('liner2_shoe', 0), 2), 'mMD'], ['Well Geometry', 'Liner 2 ID', _fmt(inputs.get('liner2_id', 0), 3), 'in']] if int(inputs.get('section_count', 2)) >= 4 else []),
        ['Drillstring / Running String', 'Operation', inputs.get('analysis_string', 'Drilling'), '—'],
        ['Drillstring / Running String', 'Drill Pipe', inputs['dp_std'], 'standard'],
        ['Drillstring / Running String', 'DP Length', _fmt(inputs['dp_length'], 2), 'm'],
        ['Drillstring / Running String', 'HWDP', inputs['hwdp_std'], 'standard'],
        ['Drillstring / Running String', 'HWDP Length', _fmt(inputs['hwdp_length'], 2), 'm'],
        ['Drillstring / Running String', 'Drill Collar', inputs['dc_std'], 'standard'],
        ['Drillstring / Running String', 'DC Length', _fmt(inputs['dc_length'], 2), 'm'],
        ['Drillstring / Running String', 'Casing / Liner', inputs.get('casing_string_spec', '—'), 'API selection'],
        ['Drillstring / Running String', 'Casing / Liner Length', _fmt(inputs.get('casing_string_length', 0), 2), 'm'],
        ['Mud Rheology', 'Mud Weight', _fmt(inputs['mud_weight'], 2), 'ppg'],
        ['Mud Rheology', 'Mud Volume in Hole', f"{_fmt(inputs.get('mud_fill_percent', 100.0), 0)}%", 'wellbore volume'],
        ['Mud Rheology', 'R600 / R300', f"{_fmt(inputs['r600'], 1)} / {_fmt(inputs['r300'], 1)}", 'dial readings'],
        ['Friction', 'Casing FF Range', f"{_fmt(inputs['ff_casing_min'], 2)} – {_fmt(inputs['ff_casing_max'], 2)}", '—'],
        ['Friction', 'Open Hole FF Range', f"{_fmt(inputs['ff_open_min'], 2)} – {_fmt(inputs['ff_open_max'], 2)}", '—'],
        ['Drilling Parameter', 'RPM', _fmt(inputs['rpm'], 0), 'rpm'],
        ['Drilling Parameter', 'WOB', _fmt(inputs['wob'], 0), 'lbf'],
        ['Drilling Parameter', 'Bit Torque', _fmt(inputs['bit_torque'], 0), 'lbf-ft'],
        ['Drilling Parameter', 'Mode', inputs['drilling_mode'], '—'],
    ]
    return rows


def _parse_standard(s):
    parts = s.replace('lb/ft', '').split('×')
    return float(parts[0].strip()), float(parts[1].strip()), float(parts[2].strip())


def _styled_table(data, widths, header_color='#444444', font_size=10, repeat_rows=1, header_font_color=colors.white):
    font_size = float(font_size)
    # Use Paragraph cells throughout so long text wraps instead of being clipped by column widths.
    cell_style = ParagraphStyle(
        name=f'TableCell_{font_size:g}', fontName='Helvetica', fontSize=font_size,
        leading=max(font_size + 1.5, font_size * 1.15), wordWrap='CJK',
        splitLongWords=1, spaceAfter=0, spaceBefore=0,
    )
    header_style = ParagraphStyle(
        name=f'TableHead_{font_size:g}', fontName='Helvetica-Bold', fontSize=font_size,
        leading=max(font_size + 1.5, font_size * 1.15), textColor=header_font_color,
        wordWrap='CJK', splitLongWords=1, spaceAfter=0, spaceBefore=0,
    )
    wrapped = []
    for ri, row in enumerate(data):
        wrapped_row = []
        for cell in row:
            if isinstance(cell, Paragraph):
                wrapped_row.append(cell)
            else:
                value = str(cell if cell is not None else '—').replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('\n', '<br/>')
                wrapped_row.append(Paragraph(value, header_style if ri == 0 else cell_style))
        wrapped.append(wrapped_row)
    t = Table(wrapped, colWidths=widths, repeatRows=repeat_rows, splitByRow=1)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor(header_color)),
        ('TEXTCOLOR', (0,0), (-1,0), header_font_color),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), font_size),
        ('GRID', (0,0), (-1,-1), 0.35, colors.HexColor('#9A9A9A')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F5F5F5')]),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    return t


def _chart_to_image(fig, dpi=150):
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf


def _axis_integer_format(ax):
    fmt = FuncFormatter(lambda x, pos: f'{x:,.0f}')
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)


def _annotate_report_depth_markers(ax, inputs):
    markers = [('KOP', inputs.get('kop')), ('EOB', inputs.get('eob')), ('Casing Shoe', inputs.get('casing_shoe'))]
    if int(inputs.get('section_count', 2)) >= 3:
        markers.append(('Liner 1 Shoe', inputs.get('liner1_shoe')))
    if int(inputs.get('section_count', 2)) >= 4:
        markers.append(('Liner 2 Shoe', inputs.get('liner2_shoe')))
    for label, depth in markers:
        if depth is None:
            continue
        depth = float(depth)
        ax.axhline(depth, color='#777777', linewidth=0.65, linestyle=(0, (3, 3)), alpha=0.8, zorder=0)
        ax.plot([0.0], [depth], marker='o', markersize=3.5, color='#444444', transform=ax.get_yaxis_transform(), clip_on=False, zorder=5)
        ax.annotate(f'{label} @{depth:,.0f} mMD', xy=(0.015, depth), xycoords=('axes fraction', 'data'),
                    xytext=(2, 2), textcoords='offset points', ha='left', va='bottom', fontsize=8,
                    color='#333333', bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.88), clip_on=True, zorder=6)


def generate_tnd_report(report_info, inputs, result_df, trajectory_df, sensitivity, icon_path=None, logo_path=None, schematic_svg=None):
    buffer = BytesIO()
    report_date = datetime.now().strftime('%d %B %Y')
    doc = SimpleDocTemplate(
        buffer, pagesize=A4, rightMargin=12*mm, leftMargin=12*mm,
        topMargin=18*mm, bottomMargin=15*mm,
        title='Preliminary Torque & Drag Analysis',
        author=report_info.get('prepared_by', 'Rigsis Drilling Team'),
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='CoverSmall', parent=styles['Normal'], alignment=TA_CENTER, fontSize=22, leading=26, textColor=colors.HexColor('#222222')))
    styles.add(ParagraphStyle(name='CoverTitle', parent=styles['Title'], alignment=TA_CENTER, fontSize=22, leading=26, textColor=colors.HexColor('#222222')))
    styles.add(ParagraphStyle(name='CoverMeta', parent=styles['BodyText'], alignment=TA_CENTER, fontSize=10, leading=14, textColor=colors.HexColor('#555555')))
    styles.add(ParagraphStyle(name='Disclaimer', parent=styles['BodyText'], alignment=TA_CENTER, fontSize=10, leading=13, textColor=colors.HexColor('#555555')))
    styles.add(ParagraphStyle(name='SectionTitle', parent=styles['Heading2'], fontSize=14, leading=17, spaceBefore=3, spaceAfter=7, textColor=colors.HexColor('#222222')))
    styles.add(ParagraphStyle(name='BodySmall', parent=styles['BodyText'], fontSize=10, leading=13))
    styles.add(ParagraphStyle(name='TableHeaderWhite', parent=styles['BodyText'], fontSize=10, leading=11, textColor=colors.white, fontName='Helvetica-Bold'))
    styles.add(ParagraphStyle(name='Method', parent=styles['BodyText'], fontSize=10, leading=14, leftIndent=5*mm, bulletIndent=2*mm))

    story = []
    _cover_page(story, report_info, inputs, icon_path, logo_path, styles)

    # 1. Report Summary
    story.append(Paragraph('1. Report Summary', styles['SectionTitle']))
    story.append(_styled_table(_summary_table(report_info, inputs, result_df), [64*mm, 102*mm], header_color='#444444', font_size=7.6))
    story.append(Spacer(1, 5*mm))
    story.append(Paragraph(
        'This report presents a preliminary J-Type 2D soft-string force-balance assessment of hookload and RPM-dependent surface torque. '
        'The results are intended for preliminary engineering screening and should be verified against detailed Torque and Drag software and field-specific data before operational use.',
        styles['BodySmall']
    ))
    story.append(PageBreak())

    # 2. Data Input
    story.append(Paragraph('2. Data Input', styles['SectionTitle']))
    story.append(_styled_table(_input_table(inputs), [45*mm, 48*mm, 67*mm, 22*mm], header_color='#444444', font_size=6.8))
    story.append(Spacer(1, 5*mm))

    # 2.1 Well Trajectory Summary starts on a fresh page.
    story.append(PageBreak())
    story.append(Paragraph('2.1 Well Trajectory Summary', styles['Heading3']))
    traj_points = []
    for md_target, label in [(0.0, 'Surface'), (inputs['kop'], 'KOP'), (inputs['eob'], 'EOB'), (inputs['td'], 'TD')]:
        idx = int(min(range(len(trajectory_df)), key=lambda i: abs(float(trajectory_df.iloc[i]['MD_m']) - float(md_target))))
        row = trajectory_df.iloc[idx]
        traj_points.append([
            label, _fmt(row['MD_m'], 0), _fmt(row['Inclination_deg'], 0), _fmt(row['Azimuth_deg'], 0),
            _fmt(row['TVD_m'], 0), _fmt(row.get('X_m', 0), 0), _fmt(row.get('Y_m', 0), 0),
        ])
    story.append(_styled_table(
        [['Point','MD (m)','Inc (°)','Azi (°)','TVD (m)','Northing (m)','Easting (m)']] + traj_points,
        [24*mm,24*mm,22*mm,22*mm,25*mm,30*mm,30*mm], header_color='#444444', font_size=7.0
    ))
    story.append(Spacer(1, 5*mm))

    # 2.2 Drillstring Arrangement
    dp_od, dp_id, dp_wt = _parse_standard(inputs['dp_std'])
    hw_od, hw_id, hw_wt = _parse_standard(inputs['hwdp_std'])
    dc_od, dc_id, dc_wt = _parse_standard(inputs['dc_std'])
    cstr = inputs.get('casing_string_spec', '—')
    cdict = None
    try:
        left, id_part = cstr.split('| ID')
        cparts = left.replace('lb/ft', '').split('×')
        cdict = [float(cparts[0].strip()), float(id_part.replace('in', '').strip()), float(cparts[1].strip())]
    except Exception:
        cdict = None

    op = inputs.get('analysis_string', 'Drilling')
    string_rows = [['Position', 'Component', 'Length (m)', 'OD (in)', 'ID (in)', 'Weight (lb/ft)']]
    if op == 'Drilling':
        string_rows += [
            ['Surface →', 'Drill Pipe', _fmt(inputs['dp_length'], 0), _fmt(dp_od,3), _fmt(dp_id,3), _fmt(dp_wt,2)],
            ['', 'HWDP', _fmt(inputs['hwdp_length'], 0), _fmt(hw_od,3), _fmt(hw_id,3), _fmt(hw_wt,2)],
            ['→ Bit', 'Drill Collar', _fmt(inputs['dc_length'], 0), _fmt(dc_od,3), _fmt(dc_id,3), _fmt(dc_wt,2)],
        ]
    elif op == 'Running Liner':
        string_rows += [
            ['Surface →', 'Drill Pipe', _fmt(inputs['dp_length'], 0), _fmt(dp_od,3), _fmt(dp_id,3), _fmt(dp_wt,2)],
            ['→ Bottom', 'Liner', _fmt(inputs.get('casing_string_length', 0), 0), _fmt(cdict[0],3), _fmt(cdict[1],3), _fmt(cdict[2],2) if cdict else '—'],
        ]
    else:
        string_rows += [
            ['Surface →', 'Casing', _fmt(inputs.get('casing_string_length', inputs['td']), 0), _fmt(cdict[0],3), _fmt(cdict[1],3), _fmt(cdict[2],2) if cdict else '—'],
        ]
    drillstring_block = [
        Paragraph('2.2 Drillstring Arrangement', styles['Heading3']),
        _styled_table(string_rows, [25*mm, 38*mm, 30*mm, 24*mm, 24*mm, 30*mm], header_color='#444444', font_size=7.2),
    ]
    story.append(KeepTogether(drillstring_block))

    # 3. Well Schematic — render the same SVG used by the Streamlit UI into the PDF.
    story.append(PageBreak())
    story.append(Paragraph('3. Well Schematic', styles['SectionTitle']))
    if schematic_svg:
        try:
            import cairosvg
            schematic_png = BytesIO(cairosvg.svg2png(bytestring=schematic_svg.encode('utf-8'), output_width=1000))
            schematic_png.seek(0)
            story.append(Image(schematic_png, width=170*mm, height=170*mm, kind='proportional', hAlign='CENTER'))
        except Exception:
            try:
                from svglib.svglib import svg2rlg
                from reportlab.graphics import renderPDF
                from reportlab.platypus import Flowable
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.svg', encoding='utf-8', delete=False) as svg_file:
                    svg_file.write(schematic_svg)
                    svg_path = svg_file.name
                drawing = svg2rlg(svg_path)
                if drawing is None:
                    raise ValueError('SVG parser returned no drawing')
                # Fit vector drawing into the printable page area, preserving aspect ratio.
                max_w, max_h = 170*mm, 220*mm
                scale = min(max_w / max(float(drawing.width), 1), max_h / max(float(drawing.height), 1))
                drawing.width *= scale
                drawing.height *= scale
                drawing.scale(scale, scale)
                story.append(drawing)
            except Exception:
                story.append(Paragraph('Unable to convert the schematic SVG for this PDF. Please check the schematic SVG data and install CairoSVG or svglib.', styles['BodySmall']))
    else:
        story.append(Paragraph('Schematic image was not supplied for this report.', styles['BodySmall']))

    # 4. Methodology
    story.append(PageBreak())
    story.append(Paragraph('4. Methodology', styles['SectionTitle']))
    method_text = [
        '<b>Well trajectory:</b> J-Type 2D trajectory using the supplied KOP, EOB and final inclination. Minimum Curvature is used to calculate the trajectory quantities used by the force-balance solver.',
        '<b>Hookload:</b> At each Run MD, the active portion of the analyzed string is solved continuously along the wellbore for Pickup, Rotating Off Bottom and Slack-off conditions. For casing or liner assessment, the selected running-string length defines the analysis depth range.',
        '<b>Force balance:</b> The preliminary model accounts for buoyed string weight, wellbore curvature, normal contact force and axial friction. The selected mud volume in hole controls the portion of the string receiving mud buoyancy. Casing and open-hole friction factors are applied according to measured depth relative to the previous casing shoe.',
        '<b>Surface torque:</b> For drillstring analysis, surface torque is represented as Bit Torque + contact torque + RPM-dependent viscous torque. Contact torque uses the relevant contact radius and the viscous component uses the Power-Law rheology with a concentric-annulus approximation.',
        '<b>Friction sensitivity:</b> Four combinations are evaluated from the specified casing and open-hole minimum/maximum friction factors. The output graphs show the resulting sensitivity envelope around the nominal midpoint case.',
        '<b>Running-string modes:</b> Drilling uses DP + HWDP + DC; Running Liner uses DP + Liner with HWDP and DC set to zero; Running Casing uses the full casing string with DP, HWDP and DC set to zero. RPM, WOB and bit torque are automatically set to zero for casing/liner running.',
        '<b>Scope:</b> This is a static preliminary model. Dynamic torsion, stick-slip, detailed beam-column behavior, motor torque and other advanced effects are outside the current scope.',
    ]
    for item in method_text:
        story.append(Paragraph('• ' + item, styles['Method']))
        story.append(Spacer(1, 2*mm))
    story.append(PageBreak())

    # 5. Output
    story.append(Paragraph('5. Output', styles['SectionTitle']))
    md = result_df['MD_m'].to_numpy()
    plot_top = min(float(result_df['MD_m'].max()), max(0.0, min(float(inputs['kop']), float(inputs['casing_shoe'])) - 100.0))
    if plot_top >= float(result_df['MD_m'].max()):
        plot_top = max(0.0, float(result_df['MD_m'].max()) - 100.0)

    sens_puw = [df['PUW_lbf'].to_numpy() for df in sensitivity.values()]
    sens_sow = [df['SOW_lbf'].to_numpy() for df in sensitivity.values()]
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    ax.fill_betweenx(md, np.min(np.vstack(sens_puw), axis=0), np.max(np.vstack(sens_puw), axis=0), alpha=0.14, label='PUW friction sensitivity')
    ax.fill_betweenx(md, np.min(np.vstack(sens_sow), axis=0), np.max(np.vstack(sens_sow), axis=0), alpha=0.10, label='SOW friction sensitivity')
    ax.plot(result_df['PUW_lbf'], md, label='PUW — nominal', linewidth=2)
    ax.plot(result_df['ROTW_lbf'], md, label='ROTW — nominal', linewidth=2)
    ax.plot(result_df['SOW_lbf'], md, label='SOW — nominal', linewidth=2)
    ax.set_ylim(float(result_df['MD_m'].max()), plot_top)
    _annotate_report_depth_markers(ax, inputs)
    ax.tick_params(axis='both', labelsize=10)
    ax.set_xlabel('Hookload (lbf)', fontsize=10); ax.set_ylabel('Measured Depth (m)', fontsize=10); ax.grid(True, alpha=0.25); ax.legend(fontsize=10)
    ax.set_title('Hookload vs Measured Depth', fontsize=12)
    _axis_integer_format(ax)
    img = _chart_to_image(fig)
    story.append(Image(img, width=155*mm, height=116*mm))
    story.append(Spacer(1, 4*mm))

    sens_torque = [df['SurfaceTorque_lbf_ft'].to_numpy() for df in sensitivity.values()]
    fig2, ax2 = plt.subplots(figsize=(7.2, 5.6))
    ax2.fill_betweenx(md, np.min(np.vstack(sens_torque), axis=0), np.max(np.vstack(sens_torque), axis=0), alpha=0.16, label='Friction sensitivity')
    ax2.plot(result_df['SurfaceTorque_lbf_ft'], md, label='Surface Torque — nominal', linewidth=2.2)
    ax2.set_ylim(float(result_df['MD_m'].max()), plot_top)
    _annotate_report_depth_markers(ax2, inputs)
    ax2.tick_params(axis='both', labelsize=10)
    ax2.set_xlabel('Surface Torque (lbf-ft)', fontsize=10); ax2.set_ylabel('Measured Depth (m)', fontsize=10); ax2.grid(True, alpha=0.25); ax2.legend(fontsize=10)
    ax2.set_title('Surface Torque vs Measured Depth', fontsize=12)
    _axis_integer_format(ax2)
    img2 = _chart_to_image(fig2)
    story.append(Image(img2, width=155*mm, height=116*mm))
    story.append(PageBreak())

    # 6. Attachment — Detailed Calculation Table
    story.append(Paragraph('6. Attachment — Detailed Calculation Table', styles['SectionTitle']))
    story.append(Paragraph('The tables below record the calculated results at each Run MD interval.', styles['BodySmall']))
    story.append(Spacer(1, 3*mm))

    main_cols = [
        ('MD_m','MD\n(m)'), ('PUW_lbf','PUW\n(lbf)'), ('ROTW_lbf','ROTW\n(lbf)'), ('SOW_lbf','SOW\n(lbf)'),
        ('SurfaceTorque_lbf_ft','Surface Torque\n(lbf-ft)'), ('BitTorque_lbf_ft','Bit Torque\n(lbf-ft)'),
        ('ContactTorque_lbf_ft','Contact Torque\n(lbf-ft)'), ('ViscousTorque_lbf_ft','Viscous Torque\n(lbf-ft)'),
    ]
    main_header = [Paragraph(c[1].replace('\\n','<br/>'), styles['TableHeaderWhite']) for c in main_cols]
    main_data = [main_header]
    for _, row in result_df.iterrows():
        main_data.append([_fmt(row[key], 0) for key, _ in main_cols])
    story.append(_styled_table(
        main_data,
        [16*mm, 19*mm, 19*mm, 19*mm, 29*mm, 23*mm, 26*mm, 27*mm],
        header_color='#444444', font_size=9.0, repeat_rows=1, header_font_color=colors.white
    ))

    story.append(Spacer(1, 5*mm))
    component_cols = [
        ('MD_m','MD\n(m)'), ('DP_in_hole_m','DP in Hole\n(m)'),
        ('HWDP_in_hole_m','HWDP in Hole\n(m)'), ('DC_in_hole_m','DC in Hole\n(m)'),
        ('Casing_in_hole_m','Casing in Hole\n(m)'), ('Liner_in_hole_m','Liner in Hole\n(m)'),
    ]
    component_header = [Paragraph(c[1].replace('\\n','<br/>'), styles['TableHeaderWhite']) for c in component_cols]
    component_data = [component_header]
    for _, row in result_df.iterrows():
        component_data.append([_fmt(row[key], 0) for key, _ in component_cols])
    story.append(_styled_table(
        component_data,
        [18*mm, 31*mm, 31*mm, 30*mm, 33*mm, 33*mm],
        header_color='#444444', font_size=9.0, repeat_rows=1, header_font_color=colors.white
    ))

    doc.build(
        story,
        onFirstPage=lambda canvas, doc: None,
        onLaterPages=_page_header_footer(logo_path, report_date, report_info.get('project_name', '')),
    )
    buffer.seek(0)
    return buffer.getvalue()
