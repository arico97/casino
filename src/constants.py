import datetime

url_base = "https://www.football-data.co.uk/"


# Make this automate on the encoding on feature engineer

mapping = {'H': 0, 'D': 1, 'A': 2}

teams_dict = {
    'Almeria': 0,
    'Sevilla': 1,
    'Sociedad': 2,
    'Las Palmas': 3,
    'Ath Bilbao': 4,
    'Celta': 5,
    'Villarreal': 6,
    'Getafe': 7,
    'Cadiz': 8,
    'Ath Madrid': 9,
    'Mallorca': 10,
    'Valencia': 11,
    'Osasuna': 12,
    'Girona': 13,
    'Barcelona': 14,
    'Betis': 15,
    'Alaves': 16,
    'Granada': 17,
    'Vallecano': 18,
    'Real Madrid': 19,
    'Valladolid': 20,
    'Espanol': 21,
    'Elche': 22,
    'Levante': 23,
    'Eibar': 24,
    'Huesca': 25,
    'Leganes': 26
    }

match_columns = ['Season']

home_team_columns = ['HomeTeam', 'HS', 'HST', 'HC', 'HF', 'HY', 'HR']

away_team_columns = [ 'AwayTeam', 'AS', 'AST', 'AC', 'AF', 'AY', 'AR']

selected_columns =  match_columns + home_team_columns + away_team_columns

checkpoint_path = "./src/models/lstm.weights.h5"

log_dir = "./src/models/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")