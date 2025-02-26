from src import FootballData, FeatureEngineer, Neural_Network, new_prediction_prep  

import pytest

@pytest.fixture
def setup_data():
    data_folder = './data'
    football_data = FootballData()
    football_data._create_df_pipeline(data_folder)
    fe = FeatureEngineer(football_data.df)
    X_train, X_test, y_train, y_test = fe._prepare_data_pipeline()
    return X_train, X_test, y_train, y_test, fe

def test_data_pipeline(setup_data):
    X_train, X_test, y_train, y_test, fe = setup_data
    assert X_train is not None
    assert X_test is not None
    assert y_train is not None
    assert y_test is not None
    assert fe is not None

def test_model_training(setup_data):
    X_train, X_test, y_train, y_test, fe = setup_data
    model = Neural_Network()
    model._train_model(X_train, X_test, y_train, y_test)
    assert model is not None

def test_prediction(setup_data):
    X_train, X_test, y_train, y_test, fe = setup_data
    model = Neural_Network()
    model._train_model(X_train, X_test, y_train, y_test)
    home_team = 'Real Madrid'
    away_team = 'Barcelona'
    Season = '24/25'
    X_pred = new_prediction_prep(fe, home_team, away_team, Season)
    prediction = model._predict(X_pred)
    assert prediction is not None
    assert 0 <= prediction <= 1