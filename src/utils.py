import pandas as pd
import sqlite3

def create_table(db_path: str, table_name: str, data: pd.DataFrame):    
    """Create a SQLite database to store the data.

    Args:
        None

    Returns:
        None
    """
    try:
        conn = sqlite3.connect(f"{db_path}/football.db")
        cursor = conn.cursor()
        columns = ', '.join([f"{col} {pd.api.types.infer_dtype(data[col])}" for col in data.columns])
        columns = columns.replace('string', 'TEXT').replace('integer', 'INTEGER').replace('floating', 'REAL')
        cursor.execute(f"""CREATE TABLE {table_name} ({columns})""")
        conn.close()
    except sqlite3.Error as e:
        print(f"Error creating the database: {e}")

def insert_data_to_db(data: pd.DataFrame, db_path: str, table_name: str):
    """Insert data into the database.

    Args:
        data: The data to insert into the database.

    Returns:
        None
    """
    try:
        conn = sqlite3.connect(f"{db_path}/football.db")
        cols = "','".join([str(i) for i in data.columns.tolist()])
        for _ , row in data.iterrows():
            sql = f"INSERT INTO {table_name} ('{cols}') VALUES ({row.to_list()})"
            conn.execute(sql)
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        print(f"Error inserting data into the database: {e}")