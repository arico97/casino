'''API for chatbot, football predictions and data scraping.'''

# TODO: create docker with previous api and dashboard
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import sqlite3
import logging

from src.sql_llm import SqlLlm
from src.scrapper import scrape_data, insert_data_to_db, create_table
from src import FootballData, FeatureEngineer, Neural_Network, new_prediction_prep

app = FastAPI()

logging.basicConfig(level=logging.INFO)

class ChatbotRequest(BaseModel):
    example_query: str
    db_path: str
    table_db: str
    data_folder: str

class FootballRequest(BaseModel):
    home_team: str
    away_team: str
    season: str
    option: int
    data_folder: str

class ScrapeRequest(BaseModel):
    country: str
    data_folder: str
    db_path: str

class DataToDbRequest(BaseModel):
    db_path: str
    table_db: str
    data_folder: str

@app.post("/chatbot")
def chatbot(request: ChatbotRequest):
    """
    Handle chatbot queries.
    """
    football_data = FootballData()
    football_data._create_df_pipeline(request.data_folder)
    logging.info(football_data.df.columns)
    conn = sqlite3.connect(request.db_path, check_same_thread=False)
    cursor = conn.cursor()
    '''
    create_table(request.db_path, request.table_db, football_data.df)
    insert_data_to_db(football_data.df, request.db_path, request.table_db)
    '''
    sql_llm = SqlLlm(conn)
    answer = sql_llm._create_answer(request.example_query)
    logging.info(answer)
    conn.close()
    return JSONResponse(content={"answer": answer})

@app.post("/football")
def football(request: FootballRequest):
    """
    Handle football predictions.
    """
    football_data = FootballData()
    football_data._create_df_pipeline(request.data_folder)
    fe = FeatureEngineer(football_data.df)
    X_train, X_test, y_train, y_test = fe._prepare_data_pipeline()
    model = Neural_Network()

    if request.option == 1:
        import mlflow
        from mlflow.tensorflow import MlflowCallback
        mlflow.tensorflow.autolog(disable=True)
        with mlflow.start_run() as run:
            model._train_model(X_train, X_test, y_train, y_test, callback=MlflowCallback(run))
    elif request.option == 2:
        model._train_model(X_train, X_test, y_train, y_test)

    X_pred = new_prediction_prep(fe, request.home_team, request.away_team, request.season)
    logging.info(X_pred)
    prediction = model._predict(X_pred)
    logging.info('prediction of model: %s', prediction)
    logging.info('quota: %s', 1/prediction)
    return JSONResponse(content={"prediction": prediction, "quota": 1/prediction})

@app.post("/scrape")
def scrape(request: ScrapeRequest):
    """
    Scrape football data.
    """
    scrape_data(request.country, request.data_folder, request.db_path)
    return JSONResponse(content={"scrape": "done"})

@app.post("/data_to_db")
def data_to_db(request: DataToDbRequest):
    """
    Insert scraped data into the database.
    """
    football_data = FootballData()
    football_data._create_df_pipeline(request.data_folder)
    create_table(request.db_path, request.table_db, football_data.df)
    insert_data_to_db(football_data.df, request.db_path, request.table_db)
    return JSONResponse(content={"data_to_db": "done"})
