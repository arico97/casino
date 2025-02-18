'''Module to load and combine football data from multiple CSV files.'''

from .join_data import get_data_files, load_dataframes, combine_dataframes  

import pandas as pd
import numpy as np 
import sqlite3



class FootballData:
    '''Class to load and manipulate football data.
    
    Attributes:
    - df (pd.DataFrame): Dataframe to store the loaded data.
    '''
    def __init__(self, df: pd.DataFrame | None = None):
        self.df = df

    def _create_df_pipeline(self, data_folder: str) -> None:
        '''Create a pipeline to load and combine dataframes.

        Arguments:
        - data_folder (str): Path to the folder containing the data files.
        '''
        files = get_data_files(data_folder)
        df_list = load_dataframes(files,data_folder)
        df = combine_dataframes(df_list)
        self.df = df
    
    def _df_to_np(self) -> np.ndarray:
        '''Convert a dataframe to a numpy array.

        Arguments:
        - df (pd.DataFrame): Dataframe to convert.

        Returns:
        - np.ndarray: Numpy array representation of the dataframe.
        '''
        return self.df.to_numpy()
    
    def _np_to_df(self, X: np.ndarray) -> pd.DataFrame:
        '''Convert a numpy array to a dataframe.

        Arguments:
        - X (np.ndarray): Numpy array to convert.

        Returns:
        - pd.DataFrame: Dataframe representation of the numpy array.
        '''
        return pd.DataFrame(X, columns=self.df.columns)
    
    def _df_to_csv(self, file_path: str, **kwargs) -> None:
        '''Save a dataframe to a CSV file.

        Arguments:
        - df (pd.DataFrame): Dataframe to save.
        - file_path (str): Path to save the CSV file.
        
        Returns:
        - None
        '''
        self.df.to_csv(file_path, index=False, **kwargs)

    def _df_to_db(self, db_path: str, table_name: str, **kwargs) -> None:
        '''Save a dataframe to a SQLite database.

        Arguments:
        - df (pd.DataFrame): Dataframe to save.
        - db_path (str): Path to save the SQLite database.
        - table_name (str): Name of the table to create.
        
        Returns:
        - None
        '''
        conn = sqlite3.connect(db_path)
        self.df.to_sql(table_name, conn, index=False, **kwargs)
        conn.close()

    
    

