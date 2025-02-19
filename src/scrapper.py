'''Scrapper module to scrape data from football-data.co.uk'''

import pandas as pd
import sqlite3
import os


from bs4 import BeautifulSoup
import requests

from io import StringIO

from constants import url_base

# TODO: Seperate db functions into a separate file
# TODO: Add columns in db: country name, competition name, all with code
def process_csv_from_url(url):
  """Process each line by removing consecutive commas from a url file.

  Args:
    url: The URL of the CSV file.

  Returns:
    A pandas DataFrame containing the processed data, or None if an error occurs.
  """
  try:
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for bad status codes

    processed_lines = []
    for line in response.text.splitlines():
      processed_line = line.replace(',,,,,', '')
      processed_lines.append(processed_line)

    # Join the processed lines back into a string
    csv_data = '\n'.join(processed_lines)

    # Create a DataFrame from the processed string
    df = pd.read_csv(StringIO(csv_data))  
    return df

  except requests.exceptions.RequestException as e:
    print(f"Error downloading file: {e}")
    return None
  except pd.errors.ParserError as e:
    print(f"Error parsing CSV data: {e}")
    return None
  except Exception as e:
    print(f"An unexpected error occurred: {e}")
    return None


def get_links(url: str):
    """Get all the links from a webpage.

    Args:
        url: The URL of the webpage.

    Returns:
        A list of links on the webpage.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        soup = BeautifulSoup(response.content, "html.parser")
        links = soup.find_all("a")
        return links
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the webpage: {e}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return []
    

def get_file_name(href: str):
    """Get the file name from a URL.

    Args:
        href: The URL of the file.

    Returns:
        The name of the file.
    """
    file_name = href.split("/")[1:3]
    file_name = [file_name[i] for i in [1,0]]
    competition = file_name[0].split('.')[0]
    file_name[0] = competition
    season1 = file_name[1][0:2]
    season2 = file_name[1][2:4]
    file_name[1] = season1 + '_' + season2
    file_name = '_'.join(file_name)
    file_name = file_name.split('.')[0]
    return file_name, season1, season2, competition

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
    except sqlite3.Error as e:
        print(f"Error creating the database: {e}")
    finally:
        conn.close()

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
    except sqlite3.Error as e:
        print(f"Error inserting data into the database: {e}")
    finally:
        conn.close()



#TODO: separate more code, save to db, https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_sql.html
# https://medium.com/analytics-vidhya/inserting-pandas-dataframes-into-database-using-insert-815f0e4e6361

def scrape_data(country: str, data_folder: str, db_path: str):
    """Scrape data from football-data.co.uk for a list of countries.

    Args:
        countries: A list of country codes to scrape data for.

    Returns:
        A dictionary containing the scraped data.
    """
    country_url = f"https://www.football-data.co.uk/{country}m.php"
    try:
        links = get_links(country_url)
        for link in links:
            href = link.get("href")
            if href and href.endswith(".csv"):
                file_name, season1, season2, competition = get_file_name(href)
                file_url = url_base + href
                print(f"Downloading: {file_name}")
                print(f"url: {file_url}")
                if file_name == '0405_E0':
                    data = process_csv_from_url(file_url)
                else:
                    data = pd.read_csv(file_url, sep = ',', encoding='cp1252')
                data['Season'] = f"{season1}/{season2}"
                data['League'] = competition
                data.rename(columns={'AS': 'AwayShots', 'HS': 'HomeShots', 
                                     'AST': 'AwayShotsTarget', 'HST': 'HomeShotsTarget',
                                       'AC': 'AwayCorners', 'HC': 'HomeCorners', 
                                       'AF': 'AwayFouls', 'HF': 'HomeFouls',
                                         'AY': 'AwayYellow', 'HY': 'HomeYellow', 
                                         'AR': 'AwayRed', 'HR': 'HomeRed'}, inplace=True)
                # data.to_csv(f'{data_folder}/{file_name}.csv', index=False)
                conn = sqlite3.connect(f"{db_path}/football.db")
                cursor = conn.cursor()
                cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{country}'")
                table_exists = cursor.fetchone()
                if table_exists:
                    insert_data_to_db(data, db_path, country)
                else:
                    create_table(db_path, country)
                    insert_data_to_db(data, db_path, country)
                conn.close()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the webpage: {e}")
    except Exception as e:
        print(f'Exception in country: {country}')
        print(f"An unexpected error occurred: {e}")