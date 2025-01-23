import streamlit as st
from db_handler import authenticate_user
import time

# Pages
def login_page():
    with st.empty().container(border=True):
        col1, _, col2 = st.columns([10,1,10])
        
        with col1:
            st.write("")
            st.write("")
            st.image("logo.png")
        
        with col2:
            st.title("Halaman Login")

            email = st.text_input("Username")
            password = st.text_input("Password", type="password")

            if st.button("Login"):
                time.sleep(2)
                if not (email and password):
                    st.error("Masukkan username dan password")
                elif authenticate_user(email, password):
                    st.session_state['authenticated'] = True
                    st.session_state['page'] = 'app'
                    st.rerun()
                else:
                    st.error("Username/Password Salah")