import preliminary
import analysis
import streamlit as st

PAGES = {
        "Preliminary Findings": findings,
        "Severity Analysis": analysis
        }

st.sidebar.title('Covid-19 County Impact')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()
