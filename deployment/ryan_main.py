import streamlit as st
from init_session import init_session
from init_session import reset_session
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter

import nltk

from wordcloud import WordCloud
from nltk.util import ngrams

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

    # Add "Semua Provinsi" option to the list
    provinsi = ["Semua Provinsi"] + list(n_uniques)

    with st.form(key='form_aq'):
        # User selects province or all provinces
        option = st.selectbox('Provinsi', provinsi)
        submitted = st.form_submit_button('Show')

    if submitted:
        # Filter dataset
        # Filtering based on options
        if option == "Semua Provinsi":
            filtered_df = df[df['company'] == company]  # No filtering by province
        else:
            filtered_df = df[(df['company'] == company) & (df['province'] == option)]

        # Ensure 'processed_reviews' column is valid
        filtered_df['processed_reviews'] = filtered_df['processed_reviews'].fillna('').astype(str)
        st.write(f"Filtered Data: {len(filtered_df)} reviews")

        if len(filtered_df) > 0:

            # Visualization 1: Topic Distribution (Pie Chart)
            st.subheader("a. Distribusi Topik")
            topic_counts = filtered_df['topic'].value_counts()

            fig1, ax1 = plt.subplots()
            ax1.pie(topic_counts, labels=topic_counts.index, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
            ax1.set_title("Distribusi Topik")
            ax1.axis('equal')  
            st.pyplot(fig1)

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

            # Visualization 2: Word Cloud
            st.subheader("b. Word Cloud")
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

            # Visualization 3: N-Grams Analysis
            st.subheader("c. Analisis N-Gram")

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

        else:
            st.write('Review tidak ditemukan')

        

        