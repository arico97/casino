import sqlite3

from src.sql_llm import SqlLlm 
# from src.scrapper import scrape_data, insert_data_to_db, create_table
from src import FootballData

example_query = "Who are the top 3 teams in SP1?"

db_path = './db/football.db'
table_db = 'football.db'
data_folder = './data'

# scrape_data('Spain', './data', db_path)


football_data = FootballData()
football_data._create_df_pipeline(data_folder)
print(football_data.df.columns)
conn = sqlite3.connect(db_path, check_same_thread=False)
cursor = conn.cursor()
'''
create_table(db_path, table_db, football_data.df)
insert_data_to_db(football_data.df, db_path, table_db)
'''
sql_llm = SqlLlm(conn)
answer = sql_llm._create_answer(example_query)
print(answer)
conn.close()