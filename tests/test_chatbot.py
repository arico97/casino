import pytest

import sqlite3

from src import FootballData
from src.sql_llm import SqlLlm


@pytest.fixture
def db_connection():
    db_path = './db/football.db'
    conn = sqlite3.connect(db_path, check_same_thread=False)
    yield conn
    conn.close()

@pytest.fixture
def football_data():
    data_folder = './data'
    football_data = FootballData()
    football_data._create_df_pipeline(data_folder)
    return football_data

def test_football_data_columns(football_data):
    expected_columns = ['column1', 'column2', 'column3']  # Replace with actual expected columns
    assert list(football_data.df.columns) == expected_columns

def test_sql_llm_answer(db_connection):
    example_query = "Who are the top 3 teams in SP1?"
    sql_llm = SqlLlm(db_connection)
    answer = sql_llm._create_answer(example_query)
    expected_answer = "Expected answer"  # Replace with the actual expected answer
    assert answer == expected_answer