import streamlit as st
import streamlit.components.v1 as components

from urllib.request import urlopen
import pandas as pd
import numpy as np

import json
import plotly.express as px
import requests
import io

def app():
    st.write("## Maps and Data")

    st.write('''
    The data in these maps represents the factors affecting the severity fo covid-19 in different counties in the United States.
    ''')


    with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
        counties = json.load(response)

    s = requests.get(covid_livedat).content
    covid_dat_org = pd.read_csv(io.StringIO(s.decode('utf-8')))

    covid_livedat = "https://raw.githubusercontent.com/nytimes/covid-19-data/master/live/us-counties.csv"
    s = requests.get(covid_livedat).content 
    covid_dat = pd.read_csv(io.StringIO(s.decode('utf-8')), converters={'fips': lambda x: str(x)})

    # covid_dat = covid_dat_org[covid_dat_org['fips'].notna()]
    # covid_dat['fips'] = covid_dat['fips'].apply(int)
    # covid_dat['fips'] = covid_dat['fips'].apply(str)

    # def fill_missing(series, limit):
    #     series = series.astype('str')
    #     series = ['0' + i if len(i) < limit else i for i in series]
    #     return series
    # covid_dat['fips'] = fill_missing(covid_dat['fips'], 5)

    covid_dat.county = covid_dat.county + " County"
    indexer = covid_dat[covid_dat.county == 'Oglala Lakota County'].index
    covid_dat.loc[indexer, 'fips'] = 46113

    fig = px.choropleth_mapbox(covid_dat, geojson=counties, locations='fips', color= 'cases',
                           hover_name = 'county',
                           range_color=(0, 33000),
                           hover_data = {'fips':False, 'cases':True},
                           color_continuous_scale="Inferno_r",
                           mapbox_style="carto-positron",
                           zoom=2.8, center = {"lat": 37.0902, "lon": -95.7129},
                           opacity=0.8,
                           labels={'cases':'Number of Cases'}
                          )

    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    # fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 750

    fig.update_layout(coloraxis_colorbar=dict(
        tickvals=[0, 5000,10000,15000,20000,25000,30000],
        ticktext=[0, "5k","10k","15k","20k","25k","30k+"],
    ))

    st.write(fig)

   fig.write_html("CovidCount.html")
   open("CovidCount.html", 'r', encoding='utf-8')

#=============================================#


    st.subheader('Number of Elderly Count Per Count') 
    elderly_count = open("ElderlyCount.html", 'r', encoding='utf-8')
    elderly_count_code = elderly_count.read()
    components.html(elderly_count_code , height=550)

#=============================================#

    st.subheader('Rate of change of COVID Cases') 
    rate_of_change = open("rate_of_change.html", 'r', encoding='utf-8')
    rate_of_change_read = rate_of_change.read()
    components.html(rate_of_change_read , height=550)



