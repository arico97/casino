'''Module for loading and combining data from CSV files.'''

import os

import pandas as pd

from typing import List


def load_dataframes(files: List, data_folder: str, **kwargs) -> List[pd.DataFrame]:
    '''Load a list of CSV files into dataframes.

    Arguments:
    - files (List): List of CSV files to load.
    - data_folder (str): Path to the folder containing the data files.

    Returns:
    - List[pd.DataFrame]: List of dataframes loaded from the CSV files.
    '''
    df_list = []
    for file in files:
        file_path = os.path.join(data_folder, file)
        try:
            df = pd.read_csv(file_path, **kwargs)
            file_info = file.split('_')
            season1 = file_info[1]
            season2 = file_info[2].split('.')[0]
            df['Season'] = f"{season1}/{season2}"
            df['League'] = file_info[0]
            df_list.append(df)
        except pd.errors.ParserError as e:
            print(f"Error reading {file}: {e}")
        except FileNotFoundError as e:
            print(f"Error: File not found {file_path}: {e}")
    return df_list

def combine_dataframes(df_list: List[pd.DataFrame], **kwargs)-> pd.DataFrame:
    '''Combine a list of dataframes into a single dataframe.

    Arguments:
    - df_list (List[pd.DataFrame]): List of dataframes to combine.

    Returns:
    - pd.DataFrame: Combined dataframe.
    '''
    combined_df = pd.concat(df_list, ignore_index=True, **kwargs)
    return combined_df

def get_data_files(data_folder: str)-> List:
    '''Get a list of CSV files in a folder.

    Arguments:
    - data_folder (str): Path to the folder containing the data files.

    Returns:
    - List: List of CSV files in the folder.
    '''
    files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
    return files

