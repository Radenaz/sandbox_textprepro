import pandas as pd
import streamlit as st

def authenticate_user(email, password):
    logindb = pd.read_csv('logdb.csv')
    check = logindb.query(f'user == "{email}" & password == "{password}"')
    ckcount = len(check)

    # Check if the user and password combination is valid
    if ckcount != 1:
        return False
    
    else:
        st.session_state['email'] = check.iloc[0]['user']
        st.session_state['company'] = check.iloc[0]['company']
        return True