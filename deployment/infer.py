import streamlit as st
from gensim.models import LdaModel
from gensim.corpora import Dictionary
import pickle
from prepro_script import text_preprocessing_id
import nest_asyncio

nest_asyncio.apply()

# Load the saved LDA model
jelek_lda_model = LdaModel.load('model_dicts/jelek_lda_model.model')

# Load the dictionary
jelek_dictionary = Dictionary.load('model_dicts/jelek_lda_dictionary.dict')

# Load the corpus (optional, for validation purposes)
with open('model_dicts/jelek_lda_corpus.pkl', 'rb') as f_jelek:
    jelek_corpus = pickle.load(f_jelek)

# Topic labels
jelek_topic_labels = {
    0: "Pelayan Buruk",
    1: "Delay / Lambat",
    2: "Miskomunikasi Kurir"
}

st.title("ExpedAnalysis - Inference")
st.write("Analyze new reviews and infer topics based on the LDA model.")

# User Input
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
                st.subheader("Results")
                st.write(f"**Original Review**: {jelek_new_review}")
                st.write(f"**Processed Review**: {jelek_processed_review}")

                st.write("**Inferred Topics with Probabilities:**")
                for topic_id, prob in topics:
                    st.write(f"  - **{jelek_topic_labels[topic_id]}**: {prob:.2%}")
    else:
        st.warning("Please enter a review before clicking Analyze.")
