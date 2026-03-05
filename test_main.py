"""Tests for main.py"""

from unittest.mock import patch, MagicMock
import io

import polars as pl
import pytest


# Minimal HTML that pandas.read_html can parse as a table matching Wikipedia's format
FAKE_WIKIPEDIA_HTML = """
<html><body>
<table>
<thead>
  <tr>
    <th>Symbol</th><th>Security</th><th>GICS Sector</th><th>GICS Sub-Industry</th>
    <th>Other</th>
  </tr>
</thead>
<tbody>
  <tr><td>AAPL</td><td>Apple Inc.</td><td>Information Technology</td><td>Technology Hardware</td><td>x</td></tr>
  <tr><td>BRK.B</td><td>Berkshire Hathaway</td><td>Financials</td><td>Multi-Sector Holdings</td><td>x</td></tr>
  <tr><td>BF.B</td><td>Brown-Forman</td><td>Consumer Staples</td><td>Distillers &amp; Vintners</td><td>x</td></tr>
</tbody>
</table>
</body></html>
"""


def _make_mock_response(html: str) -> MagicMock:
    resp = MagicMock()
    resp.text = html
    resp.raise_for_status = MagicMock()
    return resp


@patch("requests.get")
def test_get_sp500_meta_returns_polars_dataframe(mock_get):
    mock_get.return_value = _make_mock_response(FAKE_WIKIPEDIA_HTML)

    from main import get_sp500_meta
    meta = get_sp500_meta()

    assert isinstance(meta, pl.DataFrame)


@patch("requests.get")
def test_get_sp500_meta_columns(mock_get):
    mock_get.return_value = _make_mock_response(FAKE_WIKIPEDIA_HTML)

    from main import get_sp500_meta
    meta = get_sp500_meta()

    assert meta.columns == ["ticker", "company", "sector", "sub_industry"]


@patch("requests.get")
def test_get_sp500_meta_dot_to_hyphen(mock_get):
    """Ticker dots must be normalised to hyphens to match yfinance format."""
    mock_get.return_value = _make_mock_response(FAKE_WIKIPEDIA_HTML)

    from main import get_sp500_meta
    meta = get_sp500_meta()

    tickers = meta["ticker"].to_list()
    assert "BRK-B" in tickers, "BRK.B should become BRK-B"
    assert "BF-B" in tickers,  "BF.B  should become BF-B"
    assert "AAPL" in tickers,  "Plain ticker should be unchanged"

    # No dots should remain
    assert all("." not in t for t in tickers)


@patch("requests.get")
def test_get_sp500_meta_row_count(mock_get):
    mock_get.return_value = _make_mock_response(FAKE_WIKIPEDIA_HTML)

    from main import get_sp500_meta
    meta = get_sp500_meta()

    assert len(meta) == 3


@patch("requests.get")
def test_get_sp500_meta_http_error_propagates(mock_get):
    """An HTTP error from requests should propagate (raise_for_status)."""
    from requests import HTTPError

    resp = MagicMock()
    resp.raise_for_status.side_effect = HTTPError("404")
    mock_get.return_value = resp

    from main import get_sp500_meta
    with pytest.raises(HTTPError):
        get_sp500_meta()
