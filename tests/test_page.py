from __future__ import annotations

import httpx
import pytest

from taskboard_agent.page import PageFetchError, WebPageExtractor


def test_fetch_extracts_title_and_article_text() -> None:
    html = """
    <html>
      <head><title>Article title</title></head>
      <body>
        <nav>navigation</nav>
        <main>
          <article>
            <h1>Article title</h1>
            <p>これは本文です。十分な長さのある記事本文として抽出されます。</p>
            <p>追加の本文です。背景と結論を含みます。</p>
          </article>
        </main>
      </body>
    </html>
    """

    client = WebPageExtractor(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                headers={"content-type": "text/html; charset=utf-8"},
                text=html,
                request=request,
            )
        )
    )

    page = client.fetch("https://example.test/article")

    assert page.url == "https://example.test/article"
    assert page.title == "Article title"
    assert "これは本文です" in page.text


def test_fetch_rejects_non_html_content_type() -> None:
    client = WebPageExtractor(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                headers={"content-type": "application/pdf"},
                content=b"%PDF",
                request=request,
            )
        )
    )

    with pytest.raises(PageFetchError, match="HTML"):
        client.fetch("https://example.test/file.pdf")


def test_fetch_rejects_http_error() -> None:
    client = WebPageExtractor(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(404, text="not found", request=request)
        )
    )

    with pytest.raises(PageFetchError, match="HTTP 404"):
        client.fetch("https://example.test/missing")
