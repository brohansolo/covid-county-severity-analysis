import pandas as pd
from urllib.request import urlopen
import json
import requests
import io


def getCSV():

    with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
        global counties = json.load(response)

    covid_livedat = "https://raw.githubusercontent.com/nytimes/covid-19-data/master/live/us-counties.csv"
    s = requests.get(covid_livedat).content 
    global covid_dat = pd.read_csv(io.StringIO(s.decode('utf-8')), converters={'fips': lambda x: str(x)})

    covid_dat.drop(covid_dat.columns[[6,7,8,9]], axis=1, inplace = True) 

    covid_dat.county = covid_dat.county + " County"
    indexer = covid_dat[covid_dat.county == 'Oglala Lakota County'].index
    covid_dat.loc[indexer, 'fips'] = 46113

