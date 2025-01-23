import streamlit as st
from login_page import login_page
from ryan_main import app_page
from init_session import init_session, reset_session

init_session()

if st.session_state['authenticated']:
    app_page()
else:
    reset_session()
    login_page()
