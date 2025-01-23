import streamlit as st

def init_session():
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False
    if 'page' not in st.session_state:
        st.session_state['page'] = 'login'
    if 'email' not in st.session_state:
        st.session_state['email'] = ""
    if 'password' not in st.session_state:
        st.session_state['password'] = ""
    if 'company' not in st.session_state:
        st.session_state['company'] = ""
        
    

def reset_session():
    st.session_state['authenticated'] = False
    st.session_state['page'] = 'login'
    st.session_state['email'] = ""
    st.session_state['password'] = ""
    st.session_state['company'] = ""
