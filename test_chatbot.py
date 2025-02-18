import pandas as pd
import sqlite3

from src.sql_llm import SqlLlm 
from src.scrapper import scrape_data

example_query = "Who are the top 3 teams in SP1?"

db_path = './db/football.db'
table_db = 'football.db'

scrape_data('Spain', './data', db_path)

conn = sqlite3.connect(db_path, check_same_thread=False)
sql_llm = SqlLlm(conn)
answer = sql_llm._create_answer(example_query)
print(answer)
conn.close()