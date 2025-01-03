from .feature_engineer import np,pd, FeatureEngineer
from .constants import teams_dict, match_columns, away_team_columns, home_team_columns


# do a child class

def get_mean(fe, away_team, home_team, season):
    away_team_status = fe.df_filtered.loc[fe.df_filtered['AwayTeam'] == away_team,:].loc[fe.df_filtered['Season'] == season,:]
    away_team_status_mean = away_team_status.loc[:, away_team_columns+['Season']].groupby('AwayTeam').mean().reset_index()
    home_team_status = fe.df_filtered.loc[fe.df_filtered['HomeTeam'] == home_team,:].loc[fe.df_filtered['Season'] == season,:]
    home_team_status_mean = home_team_status.loc[:, home_team_columns].groupby('HomeTeam').mean().reset_index()
    match_previous_status = pd.concat([away_team_status_mean, home_team_status_mean], axis=1, join='inner')
    return match_previous_status

def new_prediction_prep(fe: FeatureEngineer, home_team: str, away_team: str, season: str) -> pd.DataFrame:
    home_encoded = teams_dict[home_team]
    away_encoded = teams_dict[away_team]
    season_encoded = fe.season_encoder.transform(np.array([season]))
    X = get_mean(fe, home_encoded, away_encoded, season_encoded[0])
    X = X.reindex(fe.scaler_order, axis= "columns")
    X_scaled = fe._scale_data(X)
    return X_scaled
