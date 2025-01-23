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
        options=["Bad Review Analysis", "Good Review Analysis", "Inference", "Logout"],
        icons=["bar-chart-line", "bar-chart-line", "search", "box-arrow-right"],
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
    if selected == "Bad Review Analysis":
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

            #############################################################
            # Visualization 1: Topic Distribution (Pie Chart)
            st.subheader("a) Distribution of Topics")
            topic_counts = filtered_df['topic'].value_counts()

            ## Create a Plotly pie chart
            fig1 = px.pie(
                topic_counts,
                values=topic_counts.values,
                names=topic_counts.index,
                title="Topic Distribution",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )

            st.plotly_chart(fig1)
            
            # Calculate counts
            delay = len(filtered_df[filtered_df['topic'] == 'Delay/ Lambat Pengiriman'])
            kurkom = len(filtered_df[filtered_df['topic'] == 'Komunikasi Kurir'])
            layan = len(filtered_df[filtered_df['topic'] == 'Kualitas Pelayan Buruk'])

            # Calculate percentages
            total = delay + kurkom + layan
            delay_pct = delay / total if total > 0 else 0
            kurkom_pct = kurkom / total if total > 0 else 0
            layan_pct = layan / total if total > 0 else 0

            # Create a dictionary to store the percentages
            percentages = {
                'keterlambatan pengiriman': delay_pct,
                'komunikasi kurir': kurkom_pct,
                'kualitas pelayanan yang buruk': layan_pct
            }

            # Filter out percentages that are 0
            filtered_percentages = {k: v for k, v in percentages.items() if v > 0}

            # Sort the dictionary by its values in descending order
            sorted_percentages = dict(sorted(filtered_percentages.items(), key=lambda item: item[1], reverse=True))

            # Extract the top three items
            top_three = list(sorted_percentages.items())[:3]

            # Dynamic words to be added to the insight
            insight_delay = "sebagian besar pelanggan di wilayah tertentu mengeluhkan waktu pengiriman yang tidak sesuai dengan ekspektasi atau janji yang diberikan."
            insight_layan = "pelayanan di gudang pada wilayah tertentu tidak memuaskan atau bahkan mengecewakan pelanggan."
            insight_kurkom = "pelanggan merasa tidak puas dengan sikap atau perilaku kurir. Hal ini dapat mencakup keluhan seperti kurir yang melempar barang, salah lokasi pengiriman, atau kurir yang sulit dihubungi."

            # Prepare the result strings
            if len(top_three) >= 1:
                if top_three[0][0] == 'keterlambatan pengiriman':
                    insight_t1 = insight_delay
                elif top_three[0][0] == 'komunikasi kurir':
                    insight_t1 = insight_kurkom
                else:
                    insight_t1 = insight_layan
                result1 = f"Dari hasil visualisasi pie chart di atas, ditemukan bahwa distribusi topik didominasi oleh {top_three[0][0]} ({top_three[0][1]*100:.1f}%). Hal ini menunjukkan bahwa {insight_t1}"
                st.write(result1)
            else:
                result1 = ""

            if len(top_three) >= 2:
                if top_three[1][0] == 'keterlambatan pengiriman':
                    insight_t2 = insight_delay
                elif top_three[1][0] == 'komunikasi kurir':
                    insight_t2 = insight_kurkom
                else:
                    insight_t2 = insight_layan
                result2 = f"Kemudian, topik selanjutnya adalah terkait {top_three[1][0]} ({top_three[1][1]*100:.1f}%), yang menunjukkan bahwa {insight_t2}"
                st.write(result2)
            else:
                result2 = ""

            if len(top_three) >= 3:
                if top_three[2][0] == 'keterlambatan pengiriman':
                    insight_t3 = insight_delay
                elif top_three[2][0] == 'komunikasi kurir':
                    insight_t3 = insight_kurkom
                else:
                    insight_t3 = insight_layan
                result3 = f"Terakhir, topik {top_three[2][0]} ({top_three[2][1]*100:.1f}%) mengindikasikan bahwa {insight_t3}"
                st.write(result3)
            else:
                result3 = ""

            #######################################################
            # Visualization 2: Word Cloud
            st.subheader("b) Word Cloud for Processed Reviews")
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(filtered_df['processed_reviews']))
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            ax2.imshow(wordcloud, interpolation='bilinear')
            ax2.axis("off")
            st.pyplot(fig2)
            
            # Tokenize the column into words
            filtered_df['words_list'] = filtered_df['processed_reviews'].apply(lambda x: str(x).split())
            
            # Flatten the list of lists and count word frequencies
            word_counts = Counter([word for words in filtered_df['words_list'] for word in words])
            
            # Create a DataFrame with the top 5 most common words
            most_common_df = pd.DataFrame(word_counts.most_common(5), columns=['word', 'count'])
            
            # Convert to list
            most_common_list = most_common_df['word'].to_list()

            # Combine into string
            mci_string = ', '.join(most_common_list[:-1]) + ' dan ' + most_common_list[-1]

            # Dynamic words to be added to the insight
            problem_delay = "keterlambatan pengiriman"
            problem_layan = "buruknya kualitas pelayanan"
            problem_kurkom = "buruknya komunikasi antara kurir dan pelanggan"
            insight2_delay = "Masalah ini menunjukkan adanya kendala dalam manajemen waktu pengiriman, yang dapat disebabkan karena rute yang kurang optimal, kurangnya armada, atau kesalahan operasional."
            insight2_layan = "Masalah ini mencerminkan ketidakpuasan pelanggan terhadap layanan di Gudang terkait."
            insight2_kurkom = "Masalah ini menunjukkan adanya kebutuhan untuk meningkatkan keterampilan komunikasi kurir dan sistem pelacakan pengiriman."

            if top_three[0][0] == 'keterlambatan pengiriman':
                problem_i2 = problem_delay
                insight2 = insight2_delay
            elif top_three[0][0] == 'komunikasi kurir':
                problem_i2 = problem_kurkom
                insight2 = insight2_kurkom
            else:
                problem_i2 = problem_layan
                insight2 = insight2_layan

            st.write(f"Berdasarkan Word Cloud diatas, dapat dilihat bahwa masalah utama yang terjadi adalah {problem_i2}. Hal ini terlihat dari kata-kata yang sering muncul, seperti {mci_string}. {insight2}")

            ##############################################
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
                
            st.write("Hasil di atas merupakan hasil kombinasi dua dan tiga kata yang paling sering digunakan dalam review pengguna.")
            
            st.subheader("d. Masukan")

            st.write("Berikut adalah masukan yang bisa kami berikan :")

            saran1 = '''
            Identifikasi Akar Permasalahan pada Keterlambatan Pengiriman :
            - Analisis alur logistik untuk menemukan bottleneck, seperti pengelolaan rute, kapasitas armada, atau penjadwalan.  
            - Terapkan teknologi optimasi rute (misalnya, sistem berbasis GPS) dan tingkatkan transparansi dengan sistem pelacakan real-time.  
            '''

            saran2 = '''
            Tingkatkan Kualitas Pelayanan di Gudang :  
            - Lakukan pelatihan intensif untuk staf gudang mengenai standar operasional dan pelayanan pelanggan.  
            - Evaluasi fasilitas gudang untuk memastikan proses penyortiran dan pemrosesan barang berjalan efisien.  
            '''

            saran3 = '''
            Perbaiki Sistem dan Komunikasi Kurir:  
            - Terapkan sistem penjadwalan komunikasi otomatis, seperti notifikasi melalui aplikasi, SMS, atau email yang memberi tahu status pengiriman.  
            - Adakan pelatihan rutin kepada kurir tentang layanan pelanggan dan penanganan barang yang baik.  
            - Sediakan feedback system khusus untuk kurir, sehingga pelanggan dapat melaporkan masalah dengan lebih mudah.  
            '''
            
            fil_unique = filtered_df['topic'].unique().tolist()

            if 'Delay/ Lambat Pengiriman' in fil_unique:
                st.markdown(saran1)
            
            if 'Kualitas Pelayan Buruk' in fil_unique:
                st.markdown(saran2)

            if 'Komunikasi Kurir' in fil_unique:
                st.markdown(saran3)

            st.markdown('''
            Monitoring dan Evaluasi Secara Berkala:  
            - Lakukan audit performa gudang dan kurir berdasarkan wilayah untuk memastikan konsistensi layanan.  
            - Adakan survei kepuasan pelanggan setelah setiap pengiriman untuk mendapatkan masukan langsung.  

            Dengan langkah-langkah tersebut, diharapkan perusahaan dapat meningkatkan efisiensi operasional, memperbaiki pengalaman pelanggan, dan memperkuat reputasi sebagai layanan ekspedisi yang andal dan memuaskan.
            ''')

    # **Page 2: Good Review Analysis**
    if selected == "Good Review Analysis":
        # Get list of provinces from bagus_labeled_documents.csv
        df2 = pd.read_csv('bagus_labeled_documents.csv')
        n_uniques = df2['province'].unique()

        # Add "All Provinces" option to the list
        provinsi2 = ["All Provinces"] + list(n_uniques)

        with st.form(key='form_aq2'):
            # User selects province or all provinces
            option2 = st.selectbox('Provinsi', provinsi2)
            submitted2 = st.form_submit_button('Show')

        if submitted2:
            # Filter dataset
            if option2 == "All Provinces":
                filtered_df2 = df2[df2['company'] == company]  # No filtering by province
            else:
                filtered_df2 = df2[(df2['company'] == company) & (df2['province'] == option2)]
            
            # Ensure 'processed_reviews' column is valid
            filtered_df2['processed_reviews'] = filtered_df2['processed_reviews'].fillna('').astype(str)
            st.write(f"Filtered Data: {len(filtered_df2)} reviews")

            #############################################################
            # Visualization 1: Topic Distribution (Pie Chart)
            st.subheader("a. Distribution of Topics")
            topic_counts2 = filtered_df2['topic'].value_counts()

            ## Create a Plotly pie chart
            fig1_2 = px.pie(
                topic_counts2,
                values=topic_counts2.values,
                names=topic_counts2.index,
                title="Topic Distribution",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )

            st.plotly_chart(fig1_2)
            
            # Calculate counts
            resp = len(filtered_df2[filtered_df2['topic'] == 'Respon Staff'])
            kacab = len(filtered_df2[filtered_df2['topic'] == 'Kantor Cabang'])
            layan2 = len(filtered_df2[filtered_df2['topic'] == 'Pelayanan Bagus'])

            # Calculate percentages
            total2 = resp + kacab + layan2
            resp_pct = resp / total2 if total2 > 0 else 0
            kacab_pct = kacab / total2 if total2 > 0 else 0
            layan2_pct = layan2 / total2 if total2 > 0 else 0

            # Create a dictionary to store the percentages
            percentages2 = {
                'pelayanan yang bagus': layan2_pct,
                'faktor kantor cabang': kacab_pct,
                'faktor respon staf': resp_pct
            }

            # Filter out percentages that are 0
            filtered_percentages2 = {k: v for k, v in percentages2.items() if v > 0}

            # Sort the dictionary by its values in descending order
            sorted_percentages2 = dict(sorted(filtered_percentages2.items(), key=lambda item: item[1], reverse=True))

            # Extract the top three items
            top_three2 = list(sorted_percentages2.items())[:3]

            # Dynamic words to be added to the insight
            insight_kacab = "sebagian besar pelanggan di wilayah tertentu mengeluhkan waktu pengiriman yang tidak sesuai dengan ekspektasi atau janji yang diberikan."
            insight_layan2 = "pelayanan di gudang pada wilayah tertentu tidak memuaskan atau bahkan mengecewakan pelanggan."
            insight_resp = "pelanggan merasa tidak puas dengan sikap atau perilaku kurir. Hal ini dapat mencakup keluhan seperti kurir yang melempar barang, salah lokasi pengiriman, atau kurir yang sulit dihubungi."

            # Prepare the result strings
            if len(top_three2) >= 1:
                if top_three2[0][0] == 'pelayanan yang bagus':
                    insight_t12 = insight_layan2
                elif top_three2[0][0] == 'faktor kantor cabang':
                    insight_t12 = insight_kacab
                else:
                    insight_t12 = insight_resp
                result12 = f"Dari hasil visualisasi pie chart di atas, ditemukan bahwa distribusi topik didominasi oleh {top_three2[0][0]} ({top_three2[0][1]*100:.1f}%). Hal ini menunjukkan bahwa {insight_t12}"
                st.write(result12)
            else:
                result12 = ""

            if len(top_three2) >= 2:
                if top_three2[1][0] == 'pelayanan yang bagus':
                    insight_t22 = insight_layan2
                elif top_three2[1][0] == 'faktor kantor cabang':
                    insight_t22 = insight_kacab
                else:
                    insight_t22 = insight_resp
                result22 = f"Kemudian, topik selanjutnya adalah terkait {top_three2[1][0]} ({top_three2[1][1]*100:.1f}%), yang menunjukkan bahwa {insight_t22}"
                st.write(result22)
            else:
                result22 = ""

            if len(top_three2) >= 3:
                if top_three2[2][0] == 'pelayanan yang bagus':
                    insight_t32 = insight_layan2
                elif top_three2[2][0] == 'faktor kantor cabang':
                    insight_t32 = insight_kacab
                else:
                    insight_t32 = insight_resp
                result32 = f"Terakhir, topik {top_three2[2][0]} ({top_three2[2][1]*100:.1f}%) mengindikasikan bahwa {insight_t32}"
                st.write(result32)
            else:
                result32 = ""

            #######################################################
            # Visualization 2: Word Cloud
            st.subheader("b) Word Cloud for Processed Reviews")
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(filtered_df2['processed_reviews']))
            fig22, ax22 = plt.subplots(figsize=(10, 5))
            ax22.imshow(wordcloud, interpolation='bilinear')
            ax22.axis("off")
            st.pyplot(fig22)
            
            # Tokenize the column into words
            filtered_df2['words_list'] = filtered_df2['processed_reviews'].apply(lambda x: str(x).split())
            
            # Flatten the list of lists and count word frequencies
            word_counts2 = Counter([word for words in filtered_df2['words_list'] for word in words])
            
            # Create a DataFrame with the top 5 most common words
            most_common_df2 = pd.DataFrame(word_counts2.most_common(5), columns=['word', 'count'])
            
            # Convert to list
            most_common_list2 = most_common_df2['word'].to_list()

            # Combine into string
            mci_string2 = ', '.join(most_common_list2[:-1]) + ' dan ' + most_common_list2[-1]

            # Dynamic words to be added to the insight
            problem_kacab = "kantor cabang yang dinilai strategis"
            problem_layan2 = "buruknya kualitas pelayanan"
            problem_staf = "faktor respon staf yang memuaskan pelanggan"
            insight2_kacab = "Masalah ini menunjukkan adanya kendala dalam manajemen waktu pengiriman, yang dapat disebabkan karena rute yang kurang optimal, kurangnya armada, atau kesalahan operasional."
            insight2_layan2 = "Masalah ini mencerminkan ketidakpuasan pelanggan terhadap layanan di Gudang terkait."
            insight2_staf = "Masalah ini menunjukkan adanya kebutuhan untuk meningkatkan keterampilan komunikasi kurir dan sistem pelacakan pengiriman."

            if top_three2[0][0] == 'faktor kantor cabang':
                problem_i22 = problem_kacab
                insight22 = insight2_kacab
            elif top_three2[0][0] == 'faktor respon staf':
                problem_i22 = problem_staf
                insight22 = insight2_staf
            else:
                problem_i22 = problem_layan2
                insight22 = insight2_layan2

            st.write(f"Berdasarkan Word Cloud diatas, dapat dilihat bahwa hal yang membuat gudang ini dinilai bagus oleh reviewer adalah {problem_i22}. Hal ini terlihat dari kata-kata yang sering muncul, seperti {mci_string2}. {insight22}")

            ##############################################
            # Visualization 3: N-Grams Analysis
            st.subheader("c) N-Grams Analysis from Processed Reviews")

            # Function to generate n-grams
            def generate_ngrams(text, n):
                tokens = nltk.word_tokenize(text)
                return list(ngrams(tokens, n))

            # Collect bigrams and trigrams
            bigrams_list2 = []
            trigrams_list2 = []

            for review2 in filtered_df2['processed_reviews']:
                bigrams_list2.extend(generate_ngrams(review2, 2))  # bi
                trigrams_list2.extend(generate_ngrams(review2, 3))  # tri

            # Count bigrams and trigrams
            bigrams_counts2 = Counter(bigrams_list2).most_common(10)
            trigrams_counts2 = Counter(trigrams_list2).most_common(10)

            # Prepare dataframes for display
            bigrams_df2 = pd.DataFrame(bigrams_counts2, columns=["Bigram", "Count"])
            trigrams_df2 = pd.DataFrame(trigrams_counts2, columns=["Trigram", "Count"])

            # Display side-by-side tables
            st.write("Top Bigrams and Trigrams:")
            col12, col22 = st.columns(2)

            with col12:
                st.subheader("Top Bigrams")
                st.table(bigrams_df2)

            with col22:
                st.subheader("Top Trigrams")
                st.table(trigrams_df2)
                
            st.write("Hasil di atas merupakan hasil kombinasi dua dan tiga kata yang paling sering digunakan dalam review pengguna.")
            
            st.subheader("d. Masukan")

            st.write("Berikut adalah masukan yang bisa kami berikan :")

            saran12 = '''
            Identifikasi Akar Permasalahan pada Keterlambatan Pengiriman :
            - Analisis alur logistik untuk menemukan bottleneck, seperti pengelolaan rute, kapasitas armada, atau penjadwalan.  
            - Terapkan teknologi optimasi rute (misalnya, sistem berbasis GPS) dan tingkatkan transparansi dengan sistem pelacakan real-time.  
            '''

            saran22 = '''
            Tingkatkan Kualitas Pelayanan di Gudang :  
            - Lakukan pelatihan intensif untuk staf gudang mengenai standar operasional dan pelayanan pelanggan.  
            - Evaluasi fasilitas gudang untuk memastikan proses penyortiran dan pemrosesan barang berjalan efisien.  
            '''

            saran32 = '''
            Perbaiki Sistem dan Komunikasi Kurir:  
            - Terapkan sistem penjadwalan komunikasi otomatis, seperti notifikasi melalui aplikasi, SMS, atau email yang memberi tahu status pengiriman.  
            - Adakan pelatihan rutin kepada kurir tentang layanan pelanggan dan penanganan barang yang baik.  
            - Sediakan feedback system khusus untuk kurir, sehingga pelanggan dapat melaporkan masalah dengan lebih mudah.  
            '''
            
            fil_unique2 = filtered_df2['topic'].unique().tolist()

            if 'Respon Staff' in fil_unique2:
                st.markdown(saran12)
            
            if 'Kantor Cabang' in fil_unique2:
                st.markdown(saran22)

            if 'Pelayanan Bagus' in fil_unique2:
                st.markdown(saran32)

            st.markdown('''
            Monitoring dan Evaluasi Secara Berkala:  
            - Lakukan audit performa gudang dan kurir berdasarkan wilayah untuk memastikan konsistensi layanan.  
            - Adakan survei kepuasan pelanggan setelah setiap pengiriman untuk mendapatkan masukan langsung.  

            Dengan langkah-langkah tersebut, diharapkan perusahaan dapat meningkatkan efisiensi operasional, memperbaiki pengalaman pelanggan, dan memperkuat reputasi sebagai layanan ekspedisi yang andal dan memuaskan.
            ''')

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
