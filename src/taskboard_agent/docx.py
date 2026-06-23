from __future__ import annotations

from io import BytesIO
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile


WORD_NAMESPACE = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
W = f"{{{WORD_NAMESPACE}}}"
MAX_DOCUMENT_XML_BYTES = 5 * 1024 * 1024
MAX_EXTRACTED_CHARS = 200_000


class DocxExtractionError(RuntimeError):
    """Raised when a DOCX cannot be safely reduced to ordered text."""


def extract_docx_text(
    content: bytes,
    *,
    max_document_xml_bytes: int = MAX_DOCUMENT_XML_BYTES,
    max_extracted_chars: int = MAX_EXTRACTED_CHARS,
) -> str:
    try:
        with ZipFile(BytesIO(content)) as archive:
            try:
                document_info = archive.getinfo("word/document.xml")
            except KeyError as exc:
                raise DocxExtractionError("DOCX is missing word/document.xml") from exc
            if document_info.file_size > max_document_xml_bytes:
                raise DocxExtractionError(
                    f"word/document.xml exceeds {max_document_xml_bytes} bytes"
                )
            document_xml = archive.read(document_info)
    except BadZipFile as exc:
        raise DocxExtractionError("attachment is not a valid DOCX ZIP package") from exc

    try:
        root = ElementTree.fromstring(document_xml)
    except ElementTree.ParseError as exc:
        raise DocxExtractionError("word/document.xml is not valid XML") from exc
    body = root.find(f".//{W}body")
    if body is None:
        raise DocxExtractionError("DOCX does not contain a document body")

    lines: list[str] = []
    _append_blocks(body, lines, depth=0)
    text = "\n".join(lines).strip()
    if not text:
        raise DocxExtractionError("DOCX contains no extractable text")
    if len(text) > max_extracted_chars:
        raise DocxExtractionError(
            f"extracted DOCX text exceeds {max_extracted_chars} characters"
        )
    return text


def _append_blocks(element: ElementTree.Element, lines: list[str], *, depth: int) -> None:
    indent = "  " * depth
    for child in element:
        if child.tag == f"{W}p":
            text = _paragraph_text(child)
            if text:
                lines.append(f"{indent}{text}")
        elif child.tag == f"{W}tbl":
            lines.append(f"{indent}[TABLE]")
            for row_index, row in enumerate(row_children(child), start=1):
                lines.append(f"{indent}  [ROW {row_index}]")
                for cell_index, cell in enumerate(cell_children(row), start=1):
                    lines.append(f"{indent}    [CELL {cell_index}]")
                    _append_blocks(cell, lines, depth=depth + 3)
        elif child.tag in {f"{W}sdt", f"{W}sdtContent", f"{W}customXml"}:
            _append_blocks(child, lines, depth=depth)


def row_children(table: ElementTree.Element) -> list[ElementTree.Element]:
    return [child for child in table if child.tag == f"{W}tr"]


def cell_children(row: ElementTree.Element) -> list[ElementTree.Element]:
    return [child for child in row if child.tag == f"{W}tc"]


def _paragraph_text(paragraph: ElementTree.Element) -> str:
    parts: list[str] = []
    for node in paragraph.iter():
        if node.tag == f"{W}t" and node.text:
            parts.append(node.text)
        elif node.tag == f"{W}tab":
            parts.append("\t")
        elif node.tag in {f"{W}br", f"{W}cr"}:
            parts.append("\n")
    return "".join(parts).strip()
