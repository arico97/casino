from src import FootballData, FeatureEngineer, Neural_Network, new_prediction_prep  

data_folder = './data'

home_team = 'Real Madrid'

away_team = 'Barcelona'

Season = '24/25'

football_data = FootballData()

football_data._create_df_pipeline(data_folder)

fe = FeatureEngineer(football_data.df)

X_train, X_test, y_train, y_test = fe._prepare_data_pipeline()

model = Neural_Network()

model._train_model(X_train, X_test, y_train, y_test)

X_pred = new_prediction_prep(fe, home_team, away_team, Season)

print(X_pred)
prediction = model._predict(X_pred)
print('prediction of model: ',prediction )
print('quota: ', 1/prediction)
