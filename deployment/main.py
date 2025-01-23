import streamlit as st
from streamlit_option_menu import option_menu
from init_session import init_session
from init_session import reset_session
from login_page import login_page
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
from collections import Counter
from wordcloud import WordCloud
from nltk.util import ngrams
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from prepro_script import text_preprocessing_id
import nltk
import pickle
import nest_asyncio
import asyncio

nest_asyncio.apply()

# Initialize session
init_session()

# Load LDA model and related files for inference
jelek_lda_model = LdaModel.load('model_dicts/jelek_lda_model.model')
jelek_dictionary = Dictionary.load('model_dicts/jelek_lda_dictionary.dict')
with open('model_dicts/jelek_lda_corpus.pkl', 'rb') as f_jelek:
    jelek_corpus = pickle.load(f_jelek)

# Define topic labels
jelek_topic_labels = {
    0: "Pelayan Buruk",
    1: "Delay / Lambat",
    2: "Miskomunikasi Kurir"
}

# App structure
def app_page():
    # st.image("", use_column_width=True)
    # Top menu with options: Analysis, Inference, Logout
    st.markdown('---')
    selected = option_menu(
        menu_title=None,
        options=["Analysis", "Inference", "Logout"],
        icons=["bar-chart-line", "search", "box-arrow-right"],
        menu_icon="list",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "#f8f9fa"},
            "icon": {"color": "black", "font-size": "16px"},
            "nav-link": {"font-size": "16px", "text-align": "center", "margin": "0px", "--hover-color": "#eee", "color": "black"},
            "nav-link-selected": {"background-color": "#2C6FFF", "color": "white"},
        },
    )

    # Handle Logout
    if selected == "Logout":
        reset_session()
        st.success("You have been logged out.")
        st.stop()

    user = st.session_state['email']
    company = st.session_state['company']
    st.title("ExpedAnalysis")
    st.write(f"Welcome, {user}!")

    # **Page 1: Analysis**
    if selected == "Analysis":
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

            # Create a Plotly pie chart
            fig1 = px.pie(
                topic_counts,
                values=topic_counts.values,
                names=topic_counts.index,
                title="Topic Distribution",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )

            st.plotly_chart(fig1)

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

    # **Page 2: Inference**
    elif selected == "Inference":
        st.subheader("Real-Time Inference")
        jelek_new_review = st.text_area("Enter a review to analyze", height=150)

        if st.button("Analyze"):
            if jelek_new_review.strip():
                with st.spinner("Processing review..."):
                    try:
                        # Preprocess the review
                        jelek_processed_review = text_preprocessing_id(jelek_new_review)
                    except Exception as e:
                        st.error(f"Error during preprocessing: {e}")
                        jelek_processed_review = None

                    if jelek_processed_review:
                        # Convert to Bag of Words
                        bow_vector = jelek_dictionary.doc2bow(jelek_processed_review.split())

                        # Get topic distribution with labels
                        topics = jelek_lda_model.get_document_topics(bow_vector, minimum_probability=0.0)

                        # Display Results
                        st.subheader("Inference Results")
                        st.write(f"**Original Review**: {jelek_new_review}")
                        st.write(f"**Processed Review**: {jelek_processed_review}")

                        st.write("**Inferred Topics with Probabilities:**")
                        for topic_id, prob in topics:
                            st.write(f"  - **{jelek_topic_labels[topic_id]}**: {prob:.2%}")
            else:
                st.warning("Please enter a review before clicking Analyze.")
