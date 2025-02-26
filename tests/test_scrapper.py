import pytest
import requests
from unittest.mock import patch, Mock
from bs4 import BeautifulSoup
import pandas as pd
from io import StringIO
from src.scrapper import process_csv_from_url, get_links, get_file_name, scrape_data

# Test process_csv_from_url
def test_process_csv_from_url_success():
    url = "http://example.com/test.csv"
    csv_content = "col1,col2\nval1,val2\nval3,val4"
    with patch('requests.get') as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = csv_content
        df = process_csv_from_url(url)
        assert df is not None
        assert df.shape == (2, 2)
        assert list(df.columns) == ["col1", "col2"]

def test_process_csv_from_url_request_exception():
    url = "http://example.com/test.csv"
    with patch('requests.get', side_effect=requests.exceptions.RequestException):
        df = process_csv_from_url(url)
        assert df is None

def test_process_csv_from_url_parser_error():
    url = "http://example.com/test.csv"
    csv_content = "col1,col2\nval1,val2\nval3,val4"
    with patch('requests.get') as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.text = csv_content
        with patch('pandas.read_csv', side_effect=pd.errors.ParserError):
            df = process_csv_from_url(url)
            assert df is None

# Test get_links
def test_get_links_success():
    url = "http://example.com"
    html_content = '<a href="link1">Link 1</a><a href="link2">Link 2</a>'
    with patch('requests.get') as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.content = html_content
        links = get_links(url)
        assert len(links) == 2
        assert links[0].get("href") == "link1"
        assert links[1].get("href") == "link2"

def test_get_links_request_exception():
    url = "http://example.com"
    with patch('requests.get', side_effect=requests.exceptions.RequestException):
        links = get_links(url)
        assert links == []

# Test get_file_name
def test_get_file_name():
    href = "data/0405/E0.csv"
    file_name, season1, season2, competition = get_file_name(href)
    assert file_name == "E0_04_05"
    assert season1 == "04"
    assert season2 == "05"
    assert competition == "E0"

# Test scrape_data
@patch('src.scrapper.get_links')
@patch('src.scrapper.process_csv_from_url')
@patch('src.scrapper.pd.read_csv')
@patch('src.scrapper.sqlite3.connect')
def test_scrape_data(mock_connect, mock_read_csv, mock_process_csv_from_url, mock_get_links):
    mock_get_links.return_value = [Mock(href="data/0405/E0.csv")]
    mock_process_csv_from_url.return_value = pd.DataFrame({"col1": ["val1"], "col2": ["val2"]})
    mock_read_csv.return_value = pd.DataFrame({"col1": ["val1"], "col2": ["val2"]})
    mock_conn = Mock()
    mock_connect.return_value = mock_conn
    mock_cursor = Mock()
    mock_conn.cursor.return_value = mock_cursor
    mock_cursor.fetchone.return_value = None

    scrape_data("england", "data_folder", "db_path")

    mock_get_links.assert_called_once()
    mock_process_csv_from_url.assert_called_once()
    mock_read_csv.assert_called_once()
    mock_connect.assert_called_once()
    mock_cursor.execute.assert_called()
    mock_conn.close.assert_called_once()