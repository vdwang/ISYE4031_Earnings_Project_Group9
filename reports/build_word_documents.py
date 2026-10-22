"""Build editable report and Meeting 1 documents from the existing analysis."""

from pathlib import Path
import re

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/docx"
NAMES = "Bill Nguyen, Sarah Wang, Cindy Le, Phoenix Moore, Victor Wang"


def configure():
    doc = Document()
    section = doc.sections[0]
    section.page_width, section.page_height = Inches(8.5), Inches(11)
    section.top_margin = section.bottom_margin = Inches(.62)
    section.left_margin = section.right_margin = Inches(.65)
    section.footer_distance = Inches(.25)
    for name in ["Normal", "Title", "Subtitle", "Heading 1", "Heading 2", "Caption"]:
        style = doc.styles[name]
        style.font.name = "Arial"
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.font.element.get_or_add_rPr().rFonts.set(qn("w:ascii"), "Arial")
        style.font.element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), "Arial")
    normal = doc.styles["Normal"]
    normal.font.size = Pt(10)
    normal.paragraph_format.line_spacing = 1.02
    normal.paragraph_format.space_after = Pt(5)
    normal.paragraph_format.widow_control = True
    doc.styles["Title"].font.size = Pt(19)
    doc.styles["Title"].paragraph_format.space_after = Pt(6)
    doc.styles["Subtitle"].font.size = Pt(10)
    doc.styles["Subtitle"].font.italic = False
    doc.styles["Subtitle"].font.bold = False
    subtitle_spacing = OxmlElement("w:spacing")
    subtitle_spacing.set(qn("w:val"), "0")
    doc.styles["Subtitle"].element.get_or_add_rPr().append(subtitle_spacing)
    doc.styles["Subtitle"].paragraph_format.space_after = Pt(7)
    for name in ["Heading 1", "Heading 2"]:
        style = doc.styles[name]
        style.font.size = Pt(12 if name == "Heading 1" else 10.5)
        style.font.bold = True
        style.paragraph_format.space_before = Pt(8)
        style.paragraph_format.space_after = Pt(4)
        style.paragraph_format.keep_with_next = True
    doc.styles["Caption"].font.size = Pt(8)
    doc.styles["Caption"].font.bold = False
    doc.styles["Caption"].font.italic = False
    doc.styles["Caption"].paragraph_format.space_after = Pt(5)
    doc.styles["Caption"].paragraph_format.keep_with_next = False
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    r = footer.add_run("Group 9 | ISYE 4031 | ")
    r.font.size = Pt(8)
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    footer._p.append(field)
    doc.core_properties.author = "ISYE 4031 Group 9"
    doc.core_properties.subject = "Signed stock returns after earnings announcements"
    doc.core_properties.comments = "Prepared with OpenAI Codex assistance"
    return doc


def text(doc, value, style=None):
    p = doc.add_paragraph(style=style)
    for i, piece in enumerate(re.split(r"\*\*(.*?)\*\*", value, flags=re.S)):
        r = p.add_run(piece)
        if i % 2:
            r.bold = True
    return p


def title(doc, value, subtitle):
    text(doc, value, "Title")
    text(doc, subtitle + "\nGeorgia Tech ISYE 4031 Fall 2026 Group 9\n" + NAMES + "\nPrepared October 21 2026", "Subtitle")


def heading(doc, value):
    value = re.sub(r"[^A-Za-z0-9 ]", " ", value)
    doc.add_heading(re.sub(r"\s+", " ", value).strip(), level=1)


def table(doc, rows, widths):
    tbl = doc.add_table(rows=0, cols=len(rows[0]))
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    for col, width in zip(tbl.columns, widths):
        col.width = Inches(width)
    for row_i, values in enumerate(rows):
        cells = tbl.add_row().cells
        for col_i, (cell, value, width) in enumerate(zip(cells, values, widths)):
            cell.width = Inches(width)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            props = cell._tc.get_or_add_tcPr()
            margins = OxmlElement("w:tcMar")
            for edge in ["top", "left", "bottom", "right"]:
                el = OxmlElement("w:" + edge)
                el.set(qn("w:w"), "65" if edge in ["top", "bottom"] else "85")
                el.set(qn("w:type"), "dxa")
                margins.append(el)
            props.append(margins)
            shading = OxmlElement("w:shd")
            shading.set(qn("w:fill"), "E0EAF2" if row_i == 0 else ("F5F5F5" if row_i % 2 else "FFFFFF"))
            props.append(shading)
            borders = OxmlElement("w:tcBorders")
            for edge in ["top", "left", "bottom", "right"]:
                border = OxmlElement("w:" + edge)
                for key, val in [("val", "single"), ("sz", "4"), ("color", "D9D9D9")]:
                    border.set(qn("w:" + key), val)
                borders.append(border)
            props.append(borders)
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.0
            if (len(values) > 3 and col_i > 1) or str(value).isdigit():
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(str(value))
            r.font.name = "Arial"
            r.font.size = Pt(8.2)
            r.bold = row_i == 0
            r.font.color.rgb = RGBColor(0, 0, 0)
        tr_pr = tbl.rows[-1]._tr.get_or_add_trPr()
        no_split = OxmlElement("w:cantSplit")
        tr_pr.append(no_split)
        if row_i == 0:
            tr_pr.append(OxmlElement("w:tblHeader"))
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(1)
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.line_spacing = Pt(2)
    p.add_run().font.size = Pt(2)
    return tbl


def figure(doc, filename, caption, width=6.65):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(1)
    p.paragraph_format.keep_with_next = True
    shape = p.add_run().add_picture(str(ROOT / "figures/milestone2" / filename), width=Inches(width))
    shape._inline.docPr.set("descr", caption)
    text(doc, caption, "Caption")


def revised_report():
    raw = (ROOT / "reports/ISYE4031_Group9_Milestone2_Word_Source.md").read_text(encoding="utf-8")
    assert "feedback" not in raw.lower()
    return raw


def build_report():
    doc = configure()
    title(doc, "Post Earnings Stock Return Analysis", "Milestone 2 Data and Exploratory Analysis")
    raw = revised_report()
    chunks = raw.split("\n\n")
    # Replace the old title and three metadata lines with the actual Word title block.
    for chunk in chunks[2:]:
        chunk = chunk.strip()
        if not chunk:
            continue
        if chunk == "<!-- page break -->":
            doc.add_page_break()
        elif chunk.startswith("## "):
            heading(doc, chunk[3:])
        elif chunk.startswith("!["):
            match = re.match(r"!\[(.*?)\]\((.*?)\)", chunk)
            figure(doc, Path(match.group(2)).name, match.group(1))
        elif chunk.startswith("Figure "):
            # Images already include their captions directly below them.
            continue
        elif chunk.startswith("|"):
            rows = [[v.strip() for v in row.strip().strip("|").split("|")] for row in chunk.splitlines()]
            rows.pop(1)
            widths = {2: [3.2, 4.0], 3: [1.15, 2.8, 3.25], 6: [1.75, 1.3, .4, 1.2, 1.3, 1.25]}[len(rows[0])]
            table(doc, rows, widths)
        elif chunk.startswith("Table 1."):
            text(doc, chunk, "Caption")
        elif chunk.startswith("**Reproducibility and references."):
            for line in chunk.splitlines():
                text(doc, line, "Caption")
        else:
            p = text(doc, chunk)
            if chunk.startswith("**AI assistance disclosure."):
                for run in p.runs:
                    run.font.size = Pt(8.5)
    (ROOT / "reports/ISYE4031_Group9_Milestone2_Word_Source.md").write_text(raw, encoding="utf-8")
    doc.core_properties.title = "Post Earnings Stock Return Analysis"
    path = OUT / "ISYE4031_Group9_Milestone2_Report.docx"
    doc.save(path)
    return path


def build_meeting1():
    doc = configure()
    title(doc, "Milestone 2 Meeting 1 Preparation", "Data pipeline and initial exploratory evidence")
    text(doc, "**Meeting purpose.** We will present the usable dataset, explain the prediction timing and data-quality concerns, and agree on the baseline and evaluation plan. The pilot contains 400 complete earnings events. Initial relationships are weak, so the main question is whether a regression can improve on a simple constant forecast in later periods.")
    text(doc, "**Meeting details.** Thursday, October 22, 2026, during 2:00-3:00 p.m. Eastern at George Tower 802 Collab. Group 9 should plan for approximately eight minutes. This packet covers the data and exploratory work for Meeting 1; the baseline results and residual diagnostics are in the full Milestone 2 report.")
    heading(doc, "Research question and intended decision")
    text(doc, "After the initial market reaction to quarterly earnings, can earnings, price, volume, momentum and volatility information predict the direction and magnitude of a stock's return over the following five trading sessions? Investors could use a successful model to inform buy, hold, sell or avoid decisions. We keep negative outcomes in the regression and do not restrict the sample to events that later rise.")
    heading(doc, "Prediction target and timing")
    text(doc, "The reaction day is the first available stock trading day strictly after the reported earnings date. We predict at that day's close. If its index is i, the outcome is P[i+5]/P[i]-1, covering sessions i+1 through i+5. Adjusted closes are used for all symbols in this snapshot. The initial reaction-day price change is a predictor and is excluded from the target. A secondary target subtracts SPY's return over the same calendar dates.")
    heading(doc, "Data sources and pipeline")
    text(doc, "One observation is one company's quarterly earnings event. Alpha Vantage EARNINGS supplies dates, EPS estimates, reported EPS and surprises. Yahoo Finance through yfinance supplies daily prices and volume for AAPL, AMZN, GOOGL, JNJ, JPM, META, MSFT, NVDA, WMT, XOM and benchmark SPY. Earnings dates span January 14, 2016-November 20, 2025. Each company contributes 40 events. Prices cover November 2, 2015-January 30, 2026. The analysis reuses existing caches; no new or paid access is required for this local snapshot.")
    table(doc, [
        ["Pipeline stage", "Observed count"],
        ["Parse cached quarterly earnings", "1,105 source records"],
        ["Restrict reported dates to 2016-2025", "400 events; 705 outside the window"],
        ["Match to first strictly later stock session", "400 matched; reject gaps over 7 calendar days"],
        ["Build features and stock and SPY targets", "400 complete events; no missingness exclusions"],
    ], [4.1, 3.1])
    heading(doc, "Predictors available at prediction time")
    text(doc, "Candidate predictors are current and previous-quarter earnings surprise (%), reported and estimated EPS (USD/share), reaction-day return, prior 5- and 20-session stock returns, prior 20-return sample volatility, prior 5-session SPY return and abnormal volume. Abnormal volume divides reaction-day volume by mean volume over the prior 20 sessions. All historical windows end before reaction day; reaction-day information is available by its close. Dollar EPS surprise remains in the dataset but is redundant with reported minus estimated EPS and is omitted from the regression.")
    doc.add_page_break()
    heading(doc, "Initial exploratory evidence")
    figure(doc, "01_return_distribution.png", "Figure 1. Signed five-session stock returns after reaction-day close. The vertical line marks zero; all 400 events remain.")
    text(doc, "Of 400 events, 223 (55.75%) have positive returns and 177 (44.25%) have negative returns; none are zero. Mean return is +0.242%, median +0.319%, and standard deviation 4.085 percentage points. Returns range from -22.196% to +13.496%. Variation is large relative to the mean, so evaluating a naive comparator and large forecast errors is important.")
    figure(doc, "02_predictor_relationships.png", "Figure 2. Earnings surprise and reaction-day return versus later signed return. Both panels retain all 400 events; lines summarize full-sample relationships.")
    text(doc, "Earnings surprise has little linear association with future return (r = 0.045). Reaction-day return has a modest positive association (r = 0.187), which is consistent with some continuation but leaves substantial scatter. These patterns motivate a simple baseline; they do not establish forecasting skill or a causal effect.")
    heading(doc, "Data quality and unusual values")
    text(doc, "There are zero missing values in the 11 candidate predictors and both return targets, zero duplicate event keys and zero invalid timing rows. Cached price series have no duplicate or invalid dates, missing close or volume fields, nonpositive closes or negative volumes. Independent reconstruction of all 400 events' dates, predictors and targets agrees within 10^-12. These checks establish internal consistency rather than historical completeness or vendor accuracy.")
    text(doc, "Seven events have absolute earnings surprise above 100%. XOM's May 1, 2020 event has a 960% surprise with estimated EPS of $0.05. We retain these source values; small denominators can create very large percentages. Later diagnostics should examine influence rather than automatically remove extreme events.")
    doc.add_page_break()
    heading(doc, "Dependence and leakage concerns")
    text(doc, "Each company has 40 repeated events. There are 597 overlapping future-window pairs across companies, and 148 events share a reaction date. Common market shocks can correlate errors; 400 rows are not 400 independent experiments. The ten currently active large companies also create a narrow sample with survivorship risk.")
    text(doc, "Date-only earnings records do not consistently distinguish before-market from after-market releases. The first-later-session rule separates the target mechanically, but it may not capture the actual initial reaction equally for every release. Retrospective EPS estimates and adjusted prices may also contain revisions unavailable at the historical prediction time. Future stock and SPY returns are used only for outcomes, never as predictors.")
    heading(doc, "Baseline and evaluation plan")
    text(doc, "Use ordinary least squares with an intercept and the 10 nonredundant predictors. The primary response is raw signed return; SPY-adjusted return is secondary. Fit the scaler and regression using 2016-2022 only (n = 280), validate on 2023-2024 (n = 80), and summarize 2025 as an exploratory test (n = 40). Remove any earlier-period event whose target window crosses the next boundary; no events require removal in this snapshot. Keep time order and do not randomly shuffle.")
    text(doc, "Compare regression forecasts with the training-mean and zero forecasts using MAE and RMSE in percentage points, with R-squared as supplementary context. Inspect residuals and influential observations. Later evaluation should use rolling chronological origins and account for company and calendar dependence. The existing baseline has already been run, so 2025 cannot be treated as a new untouched test after future model changes.")
    heading(doc, "Proposed work ownership")
    table(doc, [
        ["Member", "Proposed focus before the next meeting"],
        ["Bill Nguyen", "Review plots and error summaries by year and company"],
        ["Sarah Wang", "Review missingness, unusual values and feature definitions"],
        ["Cindy Le", "Check release timing and historical availability of data"],
        ["Phoenix Moore", "Refine application, limitations and report text"],
        ["Victor Wang", "Verify code, baseline outputs and reproducibility package"],
    ], [1.4, 5.8])
    heading(doc, "Questions for the TA")
    for question in [
        "Is the first trading day after a date-only report an acceptable reaction-day convention, or must release times be sourced?",
        "Should raw signed return remain the primary target, with SPY-adjusted return as a secondary analysis?",
        "Is the 400-event pilot sufficient for this milestone, and what historical selection rule should guide expansion?",
        "How should rolling evaluation and uncertainty account for overlapping market windows and repeated companies?",
        "Can positive signed predictions support the application without filtering out negative realized outcomes?",
    ]:
        p = text(doc, question, "List Number")
        p.paragraph_format.space_after = Pt(3)
    heading(doc, "Code and materials to bring")
    text(doc, "Bring this packet, the working project and the data-access README. Run python run_pipeline.py, then python run_milestone2.py from the project root to rebuild the analysis. The notebook notebooks/02_milestone2_analysis.ipynb exposes the outputs. Sources: Alpha Vantage EARNINGS documentation, yfinance PriceHistory documentation, and the course Milestone 2 brief. OpenAI Codex assisted with code, analysis and document preparation; the group should verify the results and follow syllabus disclosure rules.")
    doc.core_properties.title = "Milestone 2 Meeting 1 Preparation"
    path = OUT / "ISYE4031_Group9_Meeting1_Packet.docx"
    doc.save(path)
    return path


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for path in [build_report(), build_meeting1()]:
        check = Document(path)
        all_text = "\n".join(p.text for p in check.paragraphs) + "\n".join(c.text for t in check.tables for r in t.rows for c in r.cells)
        assert "feedback" not in all_text.lower()
        assert len(check.inline_shapes) == (3 if "Report" in path.name else 2)
        print(path)


if __name__ == "__main__":
    main()
