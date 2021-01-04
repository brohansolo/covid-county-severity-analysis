import streamlit as st
import io


import pandas as pd
import math
import numpy as np

# import matplotlib.pyplot as plt
# %matplotlib inline 
from sklearn.cluster import KMeans
from sklearn import datasets
from kneed import KneeLocator
import base64
from ipywidgets import HTML, interact, interactive, fixed, interact_manual, widgets, IntProgress, AppLayout, Button, Layout
# from contextlib import contextmanager

from IPython.display import display, HTML, Image, clear_output, Markdown
import plotly.express as px
# import time
import ipywidgets as widgets
from sklearn.preprocessing import MinMaxScaler
from urllib.request import urlopen
import json
import requests

def app():

    pd.options.mode.chained_assignment = None  # default='warn'

    covid_livedat = "https://raw.githubusercontent.com/nytimes/covid-19-data/master/live/us-counties.csv"
    s = requests.get(covid_livedat).content 
    covid_dat = pd.read_csv(io.StringIO(s.decode('utf-8')), converters={'fips': lambda x: str(x)})

    covid_dat.county = covid_dat.county + " County"
    indexer = covid_dat[covid_dat.county == 'Oglala Lakota County'].index
    covid_dat.loc[indexer, 'fips'] = 46113

    metrics = pd.read_csv('merged_data_final.csv', converters={'fips': lambda x: str(x)})
    merged_data = pd.merge(metrics, covid_dat, on='fips')


    merged_data['COVID Cases per Capita'] = merged_data['cases'] / merged_data['County Population']
    merged_data['COVID Deaths per Capita'] = merged_data['deaths'] / merged_data['County Population']

    norm=MinMaxScaler()
    scaled=norm.fit_transform(merged_data)
    scaled_df=pd.DataFrame(scaled,columns=merged_data.columns,index=merged_data.index)

    scaled_df.reset_index(level=0, inplace=True)
    scaled_df.reset_index(level=0, inplace=True)
    scaled_df.reset_index(level=0, inplace=True)
    scaled_df.reset_index(level=0, inplace=True)

    scaled_df.drop(columns = ['state','date'], inplace = True)
    scaled_df.set_index(['county', 'fips'], inplace = True)


    n = 19
    missing_dict = {'county': ["Slope County", "Billings County", "Oglala Lakota County", "Arthur County", 'McPherson County',
                            "Doña Ana County", "Hartley County", "Loving County", "Borden County", "McMullen County",
                            "Kenedy County", "King County", "La Salle Parish County", "Suffolk County", "Chesapeake County",
                            "Virginia Beach County", "Newport News County", "Hampton County", "Quitman County"],
                    'fips': [38087, 38007, 46113, 31005, 31117, 35013, 48205, 48301, 48033, 48311, 48261, 48269, 22059, 51800,
                            51550, 51810, 51700, 51650, 13239], 
                    'County Population':[0] * n,
                    'Elderly Count':[0] * n,
                    'Elderly per Capita':[0] * n,
                    'Maskless per Capita':[0] * n,
                    'ICU Beds':[0]*n,
                    'Rate of Change':[0]*n,
                    'COVID Cases':[0]*n,
                    'COVID Deaths':[0]*n,
                    'COVID Cases per Capita':[0] * n,
                    'COVID Deaths per Capita':[0] * n,
                    'Density per square mile':[0] * n} 

    missing_counties = pd.DataFrame(missing_dict)
    missing_counties.set_index(['county', 'fips'], inplace = True)

    scaled_df = pd.concat([scaled_df, missing_counties])

    columns = list(scaled_df.columns)
    my_dict = {k: v for v, k in enumerate(columns)}

    def get_optimal_k(subset):
        kmeans_kwargs = {
        "init": "random",
        "n_init": 10,
        "max_iter": 300,
        "random_state": 42 
        }

        # A list holds the SSE values for each k
        sse = []
        for k in range(1, 12):
            kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
            kmeans.fit(subset)
            sse.append(kmeans.inertia_)
            
        kn = KneeLocator(range(1,12), sse, curve='convex', direction='decreasing')
        
    #     plt.xlabel('k')
    #     plt.ylabel('Distortion')
    #     plt.title('The Elbow Method showing the optimal k')
    #     plt.plot(range(1,12), sse, 'bx-')
    #     plt.vlines(kn.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed', color = 'black');

        return kn.knee

    def rank(subset, rank_by):
        vals = subset.groupby('Cluster').mean()
        sorted_vals = vals.sort_values(by = rank_by)
        subset['Cluster'].replace(list(sorted_vals.index), list(vals.index), inplace = True)
        return subset

    def highlight_cols(s):
        color = 'red'
        return 'background-color: %s' % color

    def get_clusters(user_input, rank_by):
        
        a1 = list(user_input)
        a2 = list([rank_by])
        temp = list(set(a1 + a2))

        user_input = [my_dict[x] for x in temp]

        
        subset = scaled_df.iloc[:,user_input]
        merged_subset = merged_data.iloc[:,user_input]
        optimal_k = get_optimal_k(subset)
        kmeans=KMeans(n_clusters=int(optimal_k),random_state=1)
        kmeans.fit(subset)
        subset['Cluster'] = kmeans.labels_
        subset['Cluster'] = subset['Cluster'].astype(int) + 1
        
        subset = rank(subset, rank_by)
        
        merged_subset['Cluster'] = subset['Cluster']
        
        
        means = merged_subset.groupby('Cluster').mean()
        means.columns = 'Avg. ' + means.columns
        
        result = means.style.applymap(highlight_cols, subset=pd.IndexSlice[:, ['Avg. ' + rank_by]])

        display(result)
        
        merged_subset.reset_index(level=0, inplace = True)
        merged_subset.reset_index(level=1, inplace = True)
        
        sorted_df = merged_subset.sort_values(by = 'Cluster', ascending = False).reset_index(drop = True)

        subset.reset_index(level=0, inplace=True)
        subset.reset_index(level=0, inplace=True)
        
        
        
        fig = px.choropleth_mapbox(subset, geojson=counties, locations='fips', color= 'Cluster',
                            color_continuous_scale="Plasma_r",
                            hover_name = 'county',
                            hover_data = {'fips':False, 'Cluster': True},
                            mapbox_style="carto-positron",
                            zoom=2.8, center = {"lat": 37.0902, "lon": -95.7129},
                            opacity=0.8,
                            )

        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        
        
        if optimal_k == 4:
            fig.update_layout(coloraxis_colorbar=dict(
            tickvals=[1,1.5,2,2.5,3,3.5,4],
            ticktext= [1,"",2,"",3,"",4]
            ))
        else:
            fig.update_layout(coloraxis_colorbar=dict(
            tickvals=[1,1.5,2,2.5,3],
            ticktext= [1,"",2,"",3]
            ))


    #     fig.write_html("test.html")
        st.write(fig)
        
        
        
    user_input = st.multiselect('Select Variables to Use', columns) 
    rank_by = st.selectbox('Select Variable to Rank By', columns) 

    if st.button('Submit', key = '1'): 
        st.write(get_clusters(user_input, rank_by), use_column_width = True) 



    
