'''Scrapper module to scrape data from football-data.co.uk'''

import pandas as pd
import requests 

from bs4 import BeautifulSoup
import requests

from io import StringIO 

from typing import List

from constants import url_base


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
    file_name[0] = file_name[0].split('.')[0]
    file_name[1] = file_name[1][0:2]+'_'+file_name[1][2:4]
    file_name = '_'.join(file_name)
    file_name = file_name.split('.')[0]
    return file_name


#TODO: separate more code, save to db, https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_sql.html
def scrape_data(country: str):
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
                file_name = get_file_name(href)
                file_url = url_base + href
                print(f"Downloading: {file_name}")
                print(f"url: {file_url}")
                if file_name == '0405_E0':
                    data = process_csv_from_url(file_url)
                    data.to_csv(f'data/{file_name}.csv', index=False)
                else:
                    data = pd.read_csv(file_url, sep = ',', encoding='cp1252')
                    data.to_csv(f'data/{file_name}.csv', index=False)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the webpage: {e}")
    except Exception as e:
        print(f'Exception in country: {country}')
        print(f"An unexpected error occurred: {e}")