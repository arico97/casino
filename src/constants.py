import datetime

url_base = "https://www.football-data.co.uk/"


# Make this automate on the encoding on feature engineer
column_name_mapping = {
    'B365>2.5': 'Bet365Over25Goals',
    'B365<2.5': 'Bet365Under25Goals',
    'P>2.5': 'PinnacleOver25Goals',
    'P<2.5': 'PinnacleUnder25Goals',
    'Max>2.5': 'MaxOver25Goals',
    'Max<2.5': 'MaxUnder25Goals',
    'Avg>2.5': 'AverageOver25Goals',
    'Avg<2.5': 'AverageUnder25Goals',
    'BFE>2.5': 'BetfairExchangeOver25Goals',
    'BFE<2.5': 'BetfairExchangeUnder25Goals',
    '1XBCH': '1XBetHome',
    '1XBCD': '1XBetDraw',
    '1XBCA': '1XBetAway',
    'HS': 'HomeShots',
    'HST': 'HomeShotsOnTarget',
    'HC': 'HomeCorners',
    'HF': 'HomeFouls',
    'HY': 'HomeYellowCards',
    'HR': 'HomeRedCards',
    'AS': 'AwayShots',
    'AST': 'AwayShotsOnTarget',
    'AC': 'AwayCorners',
    'AF': 'AwayFouls',
    'AY': 'AwayYellowCards',
    'AR': 'AwayRedCards',
    'FTHG': 'FullTimeHomeGoals',
    'FTAG': 'FullTimeAwayGoals',
    'FTR': 'FullTimeResult',
    'HTHG': 'HalfTimeHomeGoals',
    'HTAG': 'HalfTimeAwayGoals',
    'HTR': 'HalfTimeResult',
    'Date': 'MatchDate',
    'Div': 'Division',
    'Referee': 'MatchReferee',
    'Attendance': 'MatchAttendance',
    '>2.5': 'Over25Goals',
    '<2.5': 'Under25Goals',
    'B365C>2.5': 'Bet365Over25Goals',
    'B365C<2.5': 'Bet365Under25Goals',
    'PC>2.5': 'PinnacleOver25Goals',
    'PC<2.5': 'PinnacleUnder25Goals',
    'MaxC>2.5': 'MaxOver25Goals',
    'MaxC<2.5': 'MaxUnder25Goals',
    'AvgC>2.5': 'AverageOver25Goals',
    'AvgC<2.5': 'AverageUnder25Goals',
    'BFEC>2.5': 'BetfairExchangeOver25Goals',
    'BFEC<2.5': 'BetfairExchangeUnder25Goals'
}


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