import streamlit as st
from init_session import init_session
from init_session import reset_session
from login_page import login_page
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter

import nltk

from wordcloud import WordCloud
from nltk.util import ngrams

import nest_asyncio
nest_asyncio.apply()

import asyncio

def run_async(func, *args):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(func(*args))
    loop.close()
    return result

init_session()

def app_page():
    with st.sidebar:
        if st.button("Logout"):
            reset_session()
            st.rerun()
        
    user = st.session_state['email']
    company = st.session_state['company']
    st.title("ExpedAnalysis")
    st.write(f'Selamat datang, {user}')
    
    # Get list of provinces from labeled_documents.csv
    df = pd.read_csv('labeled_documents.csv')
    n_uniques = df['province'].unique()

    # Add "All Provinces" option to the list
    provinsi = ["All Provinces"] + list(n_uniques)

    with st.form(key='form_aq'):
        # User selects province or all provinces
        option = st.selectbox('Provinsi', provinsi)
        submitted = st.form_submit_button('Show')

    if submitted:
        # Filter dataset
        if option == "All Provinces":
            filtered_df = df[df['company'] == company]  # No filtering by province
        else:
            filtered_df = df[(df['company'] == company) & (df['province'] == option)]
        
        # Ensure 'processed_reviews' column is valid
        filtered_df['processed_reviews'] = filtered_df['processed_reviews'].fillna('').astype(str)
        st.write(f"Filtered Data: {len(filtered_df)} reviews")

        # Visualization 1: Topic Distribution (Pie Chart)
        st.subheader("a) Distribution of Topics")
        topic_counts = filtered_df['topic'].value_counts()
        
        fig1, ax1 = plt.subplots()
        ax1.pie(topic_counts, labels=topic_counts.index, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
        ax1.set_title("Topic Distribution")
        ax1.axis('equal')  
        st.pyplot(fig1)

        # Visualization 2: Word Cloud
        st.subheader("b) Word Cloud for Processed Reviews")
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(filtered_df['processed_reviews']))
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.imshow(wordcloud, interpolation='bilinear')
        ax2.axis("off")
        st.pyplot(fig2)

        # Visualization 3: N-Grams Analysis
        st.subheader("c) N-Grams Analysis from Processed Reviews")

        # Function to generate n-grams
        def generate_ngrams(text, n):
            tokens = nltk.word_tokenize(text)
            return list(ngrams(tokens, n))

        # Collect bigrams and trigrams
        bigrams_list = []
        trigrams_list = []

        for review in filtered_df['processed_reviews']:
            bigrams_list.extend(generate_ngrams(review, 2))  # bi
            trigrams_list.extend(generate_ngrams(review, 3))  # tri

        # Count bigrams and trigrams
        bigrams_counts = Counter(bigrams_list).most_common(10)
        trigrams_counts = Counter(trigrams_list).most_common(10)

        # Prepare dataframes for display
        bigrams_df = pd.DataFrame(bigrams_counts, columns=["Bigram", "Count"])
        trigrams_df = pd.DataFrame(trigrams_counts, columns=["Trigram", "Count"])

        # Display side-by-side tables
        st.write("Top Bigrams and Trigrams:")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Top Bigrams")
            st.table(bigrams_df)

        with col2:
            st.subheader("Top Trigrams")
            st.table(trigrams_df)