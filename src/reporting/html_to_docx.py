"""
Turn the report page into a .docx, with nothing but the standard library.

    python3 -m src.reporting.html_to_docx report.html results/raport.docx

`python-docx` is not in the container image and cannot be added: it needs
`lxml`, whose compiled wheel does not match the image's Python. A .docx is a
ZIP of XML parts, so writing one directly costs less than fighting that, and
the result opens in Word, LibreOffice and Google Docs alike.

What survives the conversion: headings, paragraphs, lists, tables (with a
header row and borders), and bold/italic/monospace runs. Links become their
text followed by the URL in parentheses - readable on paper, and it avoids the
relationship bookkeeping that hyperlinks need. Everything visual - colours,
cards, the theme - is dropped, because a document is not a web page.
"""

import html
import re
import sys
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from typing import List, Optional

NS = ('xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
      'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"')

CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>"""

RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""

DOC_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>"""


def _style(sid: str, name: str, size: int, bold: bool, before: int, after: int,
           outline: Optional[int] = None) -> str:
    return (
        f'<w:style w:type="paragraph" w:styleId="{sid}"><w:name w:val="{name}"/>'
        f'<w:basedOn w:val="Normal"/><w:pPr>'
        f'<w:spacing w:before="{before}" w:after="{after}"/>'
        + (f'<w:outlineLvl w:val="{outline}"/>' if outline is not None else "")
        + f'</w:pPr><w:rPr>{"<w:b/>" if bold else ""}'
        f'<w:sz w:val="{size}"/></w:rPr></w:style>')


STYLES = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles {NS}>
<w:docDefaults><w:rPrDefault><w:rPr>
<w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/><w:sz w:val="22"/>
</w:rPr></w:rPrDefault></w:docDefaults>
<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/>
<w:pPr><w:spacing w:after="140" w:line="276" w:lineRule="auto"/></w:pPr></w:style>
{_style("Title", "Title", 56, True, 0, 240)}
{_style("Heading1", "heading 1", 36, True, 360, 160, 0)}
{_style("Heading2", "heading 2", 28, True, 320, 140, 1)}
{_style("Heading3", "heading 3", 24, True, 260, 120, 2)}
{_style("Caption", "caption", 18, False, 0, 200)}
<w:style w:type="paragraph" w:styleId="ListParagraph"><w:name w:val="List Paragraph"/>
<w:basedOn w:val="Normal"/><w:pPr><w:ind w:left="425"/><w:spacing w:after="80"/></w:pPr></w:style>
<w:style w:type="table" w:styleId="TableGrid"><w:name w:val="Table Grid"/>
<w:tblPr><w:tblBorders>
<w:top w:val="single" w:sz="4" w:color="BFBFBF"/><w:left w:val="single" w:sz="4" w:color="BFBFBF"/>
<w:bottom w:val="single" w:sz="4" w:color="BFBFBF"/><w:right w:val="single" w:sz="4" w:color="BFBFBF"/>
<w:insideH w:val="single" w:sz="4" w:color="BFBFBF"/><w:insideV w:val="single" w:sz="4" w:color="BFBFBF"/>
</w:tblBorders></w:tblPr></w:style>
</w:styles>"""


def esc(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


class Run:
    __slots__ = ("text", "bold", "italic", "mono")

    def __init__(self, text, bold=False, italic=False, mono=False):
        self.text, self.bold, self.italic, self.mono = text, bold, italic, mono

    def xml(self) -> str:
        props = ""
        if self.bold:
            props += "<w:b/>"
        if self.italic:
            props += "<w:i/>"
        if self.mono:
            props += '<w:rFonts w:ascii="Consolas" w:hAnsi="Consolas"/>'
        rpr = f"<w:rPr>{props}</w:rPr>" if props else ""
        return (f'<w:r>{rpr}<w:t xml:space="preserve">{esc(self.text)}</w:t></w:r>')


def para(runs: List[Run], style: str = "Normal", align: str = "") -> str:
    if not runs or not any(r.text.strip() for r in runs):
        return ""
    ppr = f'<w:pStyle w:val="{style}"/>'
    if align:
        ppr += f'<w:jc w:val="{align}"/>'
    return f"<w:p><w:pPr>{ppr}</w:pPr>{''.join(r.xml() for r in runs)}</w:p>"


class Converter(HTMLParser):
    """Enough HTML for the report: headings, text, lists and tables."""

    SKIP = {"style", "script", "title", "head"}
    # Void elements have no end tag. Counting one into `skip_depth` on the way
    # in means nothing ever counts it back out, and the rest of the document is
    # silently dropped - which is exactly what `<link>` did on the first run.
    VOID = {"link", "meta", "br", "img", "hr", "input", "source", "col"}
    HEADINGS = {"h1": "Heading1", "h2": "Heading2", "h3": "Heading3"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.body: List[str] = []
        self.runs: List[Run] = []
        self.skip_depth = 0
        self.style = "Normal"
        self.bold = self.italic = self.mono = 0
        self.href: Optional[str] = None
        # Tables are collected cell by cell and emitted whole, because a docx
        # row cannot be written before its cells are known.
        self.table: Optional[List[List[str]]] = None
        self.row: Optional[List[str]] = None
        self.cell: Optional[List[Run]] = None
        self.header_row = False

    # ------------------------------------------------------------ text runs
    def handle_data(self, data):
        if self.skip_depth:
            return
        text = re.sub(r"\s+", " ", data)
        if not text.strip() and not (self.runs or self.cell):
            return
        run = Run(text, bool(self.bold), bool(self.italic), bool(self.mono))
        (self.cell if self.cell is not None else self.runs).append(run)

    def _flush(self, style: Optional[str] = None):
        if self.runs:
            self.body.append(para(self.runs, style or self.style))
            self.runs = []

    # --------------------------------------------------------------- markup
    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP and tag not in self.VOID:
            self.skip_depth += 1
            return
        if self.skip_depth or tag in self.VOID and tag != "br":
            return
        attrs = dict(attrs)
        if tag in self.HEADINGS:
            self._flush()
            self.style = self.HEADINGS[tag]
        elif tag == "p":
            self._flush()
            self.style = "Caption" if "ro" in (attrs.get("class") or "") else "Normal"
        elif tag in ("strong", "b"):
            self.bold += 1
        elif tag in ("em", "i"):
            self.italic += 1
        elif tag == "code":
            self.mono += 1
        elif tag == "a":
            self.href = attrs.get("href")
        elif tag == "li":
            self._flush()
            self.style = "ListParagraph"
            self.runs.append(Run("• "))
        elif tag == "table":
            self._flush()
            self.table = []
        elif tag == "tr":
            self.row = []
        elif tag in ("td", "th"):
            self.cell = []
            self.header_row = tag == "th"
        elif tag == "br":
            self.handle_data(" ")

    def handle_endtag(self, tag):
        if tag in self.VOID:
            return
        if tag in self.SKIP:
            self.skip_depth = max(0, self.skip_depth - 1)
            return
        if self.skip_depth:
            return
        if tag in self.HEADINGS:
            self._flush()
            self.style = "Normal"
        elif tag in ("p", "li"):
            self._flush()
            self.style = "Normal"
        elif tag in ("strong", "b"):
            self.bold = max(0, self.bold - 1)
        elif tag in ("em", "i"):
            self.italic = max(0, self.italic - 1)
        elif tag == "code":
            self.mono = max(0, self.mono - 1)
        elif tag == "a":
            # Print the URL rather than embedding a relationship: this document
            # is read and printed, and a bare href is more useful than a
            # coloured word whose target is invisible on paper.
            if self.href and self.href.startswith("http"):
                target = self.cell if self.cell is not None else self.runs
                target.append(Run(f" ({self.href})", italic=True))
            self.href = None
        elif tag in ("td", "th"):
            if self.row is not None and self.cell is not None:
                self.row.append("".join(
                    para([Run(r.text, r.bold or self.header_row, r.italic, r.mono)
                          for r in self.cell]) for _ in [0]))
            self.cell = None
        elif tag == "tr":
            if self.table is not None and self.row:
                self.table.append(self.row)
            self.row = None
        elif tag == "table":
            if self.table:
                self.body.append(self._table_xml(self.table))
            self.table = None

    @staticmethod
    def _table_xml(rows: List[List[str]]) -> str:
        cols = max(len(r) for r in rows)
        width = int(9360 / cols)
        grid = "".join(f'<w:gridCol w:w="{width}"/>' for _ in range(cols))
        out = [f'<w:tbl><w:tblPr><w:tblStyle w:val="TableGrid"/>'
               f'<w:tblW w:w="5000" w:type="pct"/></w:tblPr>'
               f'<w:tblGrid>{grid}</w:tblGrid>']
        for n, row in enumerate(rows):
            cells = []
            for i in range(cols):
                body = row[i] if i < len(row) else ""
                shade = ('<w:shd w:val="clear" w:fill="F2F2F2"/>' if n == 0 else "")
                cells.append(f'<w:tc><w:tcPr><w:tcW w:w="{width}" w:type="dxa"/>'
                             f'{shade}</w:tcPr>{body or "<w:p/>"}</w:tc>')
            header = '<w:trPr><w:tblHeader/></w:trPr>' if n == 0 else ""
            out.append(f"<w:tr>{header}{''.join(cells)}</w:tr>")
        out.append("</w:tbl><w:p/>")
        return "".join(out)

    def result(self) -> str:
        self._flush()
        return "".join(x for x in self.body if x)


def convert(html_path: str, docx_path: str) -> None:
    source = Path(html_path).read_text(encoding="utf-8")
    # Drop the base64 image and the stylesheet before parsing: neither survives
    # into a document, and the data URI is 128 KB of noise.
    source = re.sub(r"<style.*?</style>", "", source, flags=re.S)
    source = re.sub(r"<img[^>]*>", "", source)
    title = re.search(r"<title>(.*?)</title>", source, flags=re.S)

    conv = Converter()
    conv.feed(source)
    body = conv.result()
    if title:
        body = para([Run(html.unescape(title.group(1)))], "Title") + body

    document = (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                f'<w:document {NS}><w:body>{body}'
                f'<w:sectPr><w:pgSz w:w="11906" w:h="16838"/>'
                f'<w:pgMar w:top="1134" w:right="1134" w:bottom="1134" w:left="1134"/>'
                f'</w:sectPr></w:body></w:document>')

    Path(docx_path).parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(docx_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", CONTENT_TYPES)
        z.writestr("_rels/.rels", RELS)
        z.writestr("word/_rels/document.xml.rels", DOC_RELS)
        z.writestr("word/styles.xml", STYLES)
        z.writestr("word/document.xml", document)
    size = Path(docx_path).stat().st_size
    print(f"✓ {docx_path} ({size / 1024:.0f} KB)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__.strip().splitlines()[2].strip())
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])
