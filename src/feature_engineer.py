'''Module for data preprocessing and feature engineering.'''

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder

from .constants import mapping, selected_columns, teams_dict



class FeatureEngineer:
    '''Handle data preprocessing steps including encoding, feature selection, scaling, and reshaping.''' 
    
    def __init__(self, df: pd.DataFrame):
        '''Initialize encoders, scalers, and data attributes.

        Arguments:
        - df (pd.DataFrame): The input dataframe to be processed.

        Attributes:
        - season_encoder (LabelEncoder): Encodes the 'Season' feature.
        - scaler (StandardScaler): Scales numerical features.
        - scaler_order (list): Tracks the order of features for scaling.
        - df (pd.DataFrame): Stores the input dataframe.
        - df_encoded (pd.DataFrame): Stores the dataframe after encoding.
        - df_filtered (pd.DataFrame): Stores the dataframe after feature selection.
        '''
        self.season_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.scaler_order = None
        self.df = df 
        self.df_encoded = None
        self.df_filtered = None

    def _encode_features(self, away_team: str = 'AwayTeam', home_team: str = 'HomeTeam') -> None:
        '''Encode categorical features in the dataframe.

        Arguments:
        - away_team (str): Column name for the away team. Defaults to 'AwayTeam'.
        - home_team (str): Column name for the home team. Defaults to 'HomeTeam'.

        Attributes Updated:
        - df_encoded (pd.DataFrame): Encoded dataframe with mapped team and season values.

        Returns:
        - None
        '''
        df = self.df.copy()
        df['Season'] = self.season_encoder.fit_transform(df['Season'])
        df[home_team] = df[home_team].map(teams_dict)
        df[away_team] = df[away_team].map(teams_dict)
        df['FTR_encoded'] = df['FTR'].map(mapping)
        self.df_encoded = df

    def _select_features(self) -> None:
        '''Select relevant features based on predefined columns and data availability.

        Attributes Accessed:
        - df_encoded (pd.DataFrame): Encoded dataframe.

        Attributes Updated:
        - df_filtered (pd.DataFrame): Filtered dataframe with selected features.

        Returns:
        - None
        '''
        data = self.df_encoded[selected_columns+['FTR_encoded']]
       # data_description = data.describe().transpose()
        # features = data_description.loc[data_description['count'] ==1679,:].index.values.tolist()
        # data_filtered = data[features].dropna()
       # self.df_filtered =  data_filtered
        self.df_filtered = data.dropna()
        print(self.df_filtered)

    def _split_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        '''Split the filtered data into training and testing sets.

        Attributes Accessed:
        - df_filtered (pd.DataFrame): Filtered dataframe after feature selection.

        Attributes Updated:
        - scaler_order (list): Saves the column order for scaling.

        Returns:
        - X_train (pd.DataFrame): Training features.
        - X_test (pd.DataFrame): Testing features.
        - y_train (pd.Series): Training target labels.
        - y_test (pd.Series): Testing target labels.
        '''
        X = self.df_filtered.drop(columns=['FTR_encoded'], axis=1)
        y = self.df_filtered['FTR_encoded']
        self.scaler_order = X.columns
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
        return X_train, X_test, y_train, y_test
    
    def _fit_scaler(self, X: pd.DataFrame) -> np.ndarray:
        '''Fit scaler to provided data and return scaled values.

        Arguments:
        - X (pd.DataFrame): The input features to fit the scaler on.

        Returns:
        - np.ndarray: Scaled feature values.
        '''
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled

    def _scale_data(self, X: pd.DataFrame) -> np.ndarray:
        '''Apply pre-fitted scaler to provided data.

        Arguments:
        - X (pd.DataFrame): The input features to be scaled.

        Returns:
        - np.ndarray: Scaled feature values.
        '''
        X = self.scaler.transform(X)
        return X

    def _reshape_data(self, data: np.ndarray) -> np.ndarray:
        '''Reshape data for LSTM model compatibility.

        Arguments:
        - data (np.ndarray): The data to reshape.

        Returns:
        - np.ndarray: Reshaped data with an additional dimension.
        '''
        reshaped_data = data.reshape((data.shape[0], 1, data.shape[1]))
        return reshaped_data

    def _prepare_data_pipeline(self, reshape: bool=False, **kwargs) -> tuple:
        '''Prepare full data processing pipeline including encoding, scaling, and reshaping.

        Arguments:
        - reshape (bool): Whether to reshape the data for LSTM input. Defaults to True.
        - kwargs: Additional keyword arguments for encoding features.

        Returns:
        - tuple: Processed training and testing sets in the format (X_train, X_test, y_train, y_test).
        '''
        self._encode_features(**kwargs)
        self._select_features()
        X_train, X_test, y_train, y_test = self._split_data()
        X_train_scaled = self._fit_scaler(X_train)
        X_test_scaled = self._scale_data(X_test)
        if reshape:
            X_train_reshaped = self._reshape_data(X_train_scaled)
            X_test_reshaped = self._reshape_data(X_test_scaled)
            return X_train_reshaped, X_test_reshaped, y_train, y_test
        return X_train_scaled, X_test_scaled, y_train, y_test