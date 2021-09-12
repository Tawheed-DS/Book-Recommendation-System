
import pickle
import pandas as pd
import streamlit as st
import base64
from sklearn.neighbors import NearestNeighbors

def main():
    st.set_page_config(layout="wide")
    st.title(" Recommendation Project")

    menu = ["About us","Recommend","Special Thanks"]
    choice = st.sidebar.selectbox("Menu",menu)

    if choice == "About us":
        st.subheader("Our Team Members")
        main_bg = "img_4.png"
        main_bg_ext = "jpg"
        st.markdown(
            f"""
            <style>
            .reportview-container {{
                background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
        st.write("The Team Space has been involved in a lot of quality project assignments."
                 "Our team works tirelessly to give the best results to the users."
                 "We take pride in our collaborative team work whether it comes to coding, assignments or presentation."
                 "Let's look at the details of our team members")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.image("Saubhagya_verma.png")
            st.text("NAME : Saubhagya Verma")
            st.text("Email_ID : saubhagyav3@gmail.com")
        with col2:
            st.image("Harsh_mudgil.png")
            st.text("NAME : Harsh Mudgil")
            st.text("Email_ID : harshmudgil72@gmail.com ")
        with col3:
            st.image("Harshal.png")
            st.text("NAME : Harshal Pawar")
            st.text("Email_ID : harshaljpawar88@gmail.com")
        with col4:
            st.image("Tawheed.png")
            st.text("NAME : Tawheed yousuf")
            st.text("Email_ID : mtawheedyousuf@gmail.com")

    elif choice == "Recommend":
        import time
        my_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.1)
            my_bar.progress(percent_complete + 1)

        main_bg = "img_4.png"
        main_bg_ext = "jpg"
        st.markdown(
            f"""
            <style>
            .reportview-container {{
                background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
        book_list = pickle.load(open('df_final.pkl', 'rb'))
        book_pics = pickle.load(open('df_final.pkl', 'rb'))
        csr_matrix = pickle.load(open('condensed_matrix.pkl', 'rb'))
        sparse_matrix = pickle.load(open('sparsed_matrix.pkl', 'rb'))

        def fetch_poster(recommended_book_name, book_pics):
            link = book_pics[book_pics['new_title'] == recommended_book_name]['Image-URL-L'].tolist()
            return link[0]

        def recommend(selected_book, csr_matrix, n_values=7):
            model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=5)
            model_knn.fit(csr_matrix)
            distances, indices = model_knn.kneighbors(sparse_matrix.loc[selected_book, :].values.reshape(1, -1),
                                                      n_neighbors=n_values)
            recommended_book_names = []
            for i in range(0, len(distances.flatten())):
                recommended_book_names.append(sparse_matrix.index[indices.flatten()[i]])
            recommended_book_names.pop(0)
            return recommended_book_names
        st.header('Book Recommender System')
        book_list = sparse_matrix.index.tolist()

        selected_book = st.selectbox(
            " Select a Particular Book to get Recommendations",
            book_list)
        if st.button('Show Recommendation'):
            recommended_book_names = recommend(selected_book, csr_matrix)
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            with col1:
                st.text(recommended_book_names[0])
                st.image(fetch_poster(recommended_book_names[0], book_pics))
            with col2:
                st.text(recommended_book_names[1])
                st.image(fetch_poster(recommended_book_names[1], book_pics))

            with col3:
                st.text(recommended_book_names[2])
                st.image(fetch_poster(recommended_book_names[2], book_pics))
            with col4:
                st.text(recommended_book_names[3])
                st.image(fetch_poster(recommended_book_names[3], book_pics))
            with col5:
                st.text(recommended_book_names[4])
                st.image(fetch_poster(recommended_book_names[4], book_pics))
            with col6:
                st.text(recommended_book_names[5])
                st.image(fetch_poster(recommended_book_names[5], book_pics))

    else:
        st.subheader("Built Under Alma Better Program")
        st.image('Alma_Better.png', width=400)
        st.text("Built with Streamlit & Pandas")

        st.write(' Our team was provided with this project last monday. '
                 'we were really excited to built our very own recommendation system. '
                 'In the Last few days we have gone through a lot of approaches used in recommendation systems. '
                 'This recommendation system is brought to you by Team Space from Alma Better. ' 
                 ' Special thanks to our team mentor Ekta Maheshwarii for guiding us in this project. '
                 'Hi Richa!, We hope you like our work! ')

        col1, col2 = st.columns(2)
        with col1:
            st.text("Ekta Maheshwarii\n"
                    "(Our Mentor)")
            st.image("Ekta.png", width=100)
        with col2:
            st.text("Richa Sharma\n"
                    "(Our Evaluator)")
            st.image("Richa.png", width=100)



if __name__ == '__main__':
	main()














