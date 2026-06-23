from __future__ import annotations

from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from taskboard_agent.docx import DocxExtractionError, extract_docx_text


def _docx(document_xml: str) -> bytes:
    output = BytesIO()
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", document_xml)
    return output.getvalue()


def test_extract_docx_text_preserves_paragraph_and_nested_table_order() -> None:
    xml = """<?xml version="1.0" encoding="UTF-8"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:r><w:t>冒頭</w:t></w:r></w:p>
    <w:tbl><w:tr><w:tc>
      <w:p><w:r><w:t>案件</w:t></w:r></w:p>
      <w:tbl><w:tr><w:tc><w:p><w:r><w:t>進捗70%</w:t></w:r></w:p></w:tc></w:tr></w:tbl>
      <w:p><w:r><w:t>次週予定</w:t></w:r></w:p>
    </w:tc></w:tr></w:tbl>
    <w:p><w:r><w:t>末尾</w:t></w:r></w:p>
  </w:body>
</w:document>"""

    text = extract_docx_text(_docx(xml))

    assert text.index("冒頭") < text.index("案件")
    assert text.index("案件") < text.index("進捗70%") < text.index("次週予定")
    assert text.index("次週予定") < text.index("末尾")
    assert text.count("進捗70%") == 1


def test_extract_docx_text_rejects_invalid_or_oversized_input() -> None:
    with pytest.raises(DocxExtractionError, match="valid DOCX"):
        extract_docx_text(b"not a zip")

    xml = """<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>長い本文</w:t></w:r></w:p></w:body></w:document>"""
    with pytest.raises(DocxExtractionError, match="exceeds 2 characters"):
        extract_docx_text(_docx(xml), max_extracted_chars=2)
