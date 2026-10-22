"""Unit tests for TableParser."""

from bs4 import BeautifulSoup

from src.parse.table_parser import TableParser


def test_clean_cell_text():
    html = "<td>  Revenue &amp; Sales&#160; <span>(in millions)</span> | </td>"
    soup = BeautifulSoup(html, "html.parser")
    cleaned = TableParser.clean_cell_text(soup.td)
    assert cleaned == "Revenue & Sales (in millions) \\|"


def test_table_colspan_rowspan():
    html = """
    <table>
      <tr>
        <th rowspan="2">Metric</th>
        <th colspan="2">Period</th>
      </tr>
      <tr>
        <th>2023</th>
        <th>2022</th>
      </tr>
      <tr>
        <td>Sales</td>
        <td>100</td>
        <td>90</td>
      </tr>
    </table>
    """
    soup = BeautifulSoup(html, "html.parser")
    parser = TableParser()
    matrix = parser.table_to_matrix(soup.table)
    assert len(matrix) == 3
    assert len(matrix[0]) == 3
    assert matrix[0][0] == "Metric"
    assert matrix[0][1] == "Period"
    assert matrix[1][1] == "2023"
    assert matrix[1][2] == "2022"
    assert matrix[2][0] == "Sales"
    assert matrix[2][1] == "100"
    assert matrix[2][2] == "90"


def test_prune_empty_and_merge_currency():
    html = """
    <table>
      <tr>
        <th>Metric</th>
        <th></th>
        <th>Amount</th>
      </tr>
      <tr>
        <td>Net Income</td>
        <td>$</td>
        <td>50,000</td>
      </tr>
      <tr>
        <td>Expenses</td>
        <td>$</td>
        <td>30,000</td>
      </tr>
    </table>
    """
    soup = BeautifulSoup(html, "html.parser")
    parser = TableParser()
    md = parser.parse_table(soup.table)
    assert "$50,000" in md
    assert "$30,000" in md
    assert "| Metric | Amount |" in md


def test_table_with_ixbrl_tags():
    html = """
    <table>
      <tr><th>Line Item</th><th>Value</th></tr>
      <tr>
        <td>Revenues</td>
        <td><ix:nonfraction unitref="USD" decimals="-6">12,500</ix:nonfraction></td>
      </tr>
    </table>
    """
    soup = BeautifulSoup(html, "html.parser")
    parser = TableParser()
    md = parser.parse_table(soup.table)
    assert "12,500" in md
    assert "Revenues" in md
