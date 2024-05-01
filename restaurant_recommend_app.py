import pickle
import streamlit as st
import pandas as pd 
import numpy as np
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("Let's help you decide where to eat today!")

#Get user to select the desired location

location_list=['Indiranagar', 'Koramangala', 'St. Marks Road', 'Cunningham Road',
       'Richmond Road', 'Lavelle Road', 'MG Road', 'Brigade Road',
       'Marathahalli', 'Hennur', 'Jayanagar', 'Sarjapur',
       'Old Airport Road', 'JP Nagar', 'Whitefield', 'Bannerghatta',
       'Hosur Road', 'BTM', 'Kalyan Nagar', 'Ashok Nagar',
       'Church Street', 'Basavanagudi', 'Malleshwaram', 'HSR',
       'Brookefield', 'Ulsoor', 'Yelahanka', 'Domlur', 'Hebbal',
       'Sahakara Nagar', 'New BEL Road', 'Nagarbhavi', 'Bhartiya City',
       'Rajarajeshwari Nagar', 'Old Madras Road', 'Kanakapura',
       'Bellandur', 'Electronic City', 'Race Course Road', 'Nagawara',
       'Vasanth Nagar']

user_location = st.selectbox(
    'Which area would you like to visit today',
    tuple(location_list))

st.write('You selected:', user_location)

#Get user cuisine preference

cuisine_list = ['south indian', 'asian', 'thai', 'continental', 'finger food', 'north indian', 'pasta', 'pizza', 'desserts', 'beverages', 'coffee cafe',
       'american', 'italian', 'chinese', 'biryani', 'tea', 'european',
    'bar', 'oriental', 'bbq', 'mughlai', 'kebab',
       'kashmiri', 'seafood', 'grilled chicken', 'fast food', 'salad',
       'street food', 'sushi', 'lebanese', 'greek', 'moroccan',
       'indonesian', 'andhra', 'mediterranean', 'japanese', 'burger',
       'sandwich', 'mangalorean', 'cantonese', 'korean', 'mexican',
       'sichuan', 'bakery', 'burmese', 'bubble tea', 'ice cream',
       'tex-mex', 'steak', 'shake', 'healthy', 'panini', 'rolls',
       'malaysian', 'hyderabadi', 'kerala', 'vietnamese', 'pancake',
       'goan', 'lucknowi', 'parsi', 'turkish', 'middle eastern',
       'gujarati', 'rajasthani', 'brazilian', 'momos', 'wraps', 'french',
       'juices', 'awadhi', 'mithai', 'portuguese', 'arabian', 'peruvian',
       'waffle', 'shawarma', 'mandi', 'nepalese', 'irish', 'british',
       'bengali', 'afghan']

user_cuisine = st.selectbox(
    'Which cuisine would you like to try today',
    tuple(cuisine_list))

st.write('You selected:', user_cuisine)

def input_transformer(location, cuisine):
    location=str(location)
    cuisine=str(cuisine)
    location=location.replace(' ','')
    user_input = location + ' ' + cuisine
    final_output=('').join(user_input.lower().split('  '))
    return final_output

preprocessed_input = input_transformer(user_location, user_cuisine)
query_tokens=preprocessed_input.split()

#st.button("Reset", type="primary")
if st.button('Generate Recommendations'):
    
    # Load the object from the pickle file
    with open('word_vectors.pkl', 'rb') as f:
        word_vectors = pickle.load(f)
        
    # Load the object from the pickle file
    with open('document_vectors.pkl', 'rb') as f:
        document_vectors = pickle.load(f)

    # Check if each word in the query exists in the vocabulary of the Word2Vec model
    valid_tokens = [word for word in query_tokens if word in word_vectors]

    if valid_tokens:
    # Calculate the average vector representation of the query
        query_vector = np.mean([word_vectors[word] for word in valid_tokens], axis=0).reshape(1, -1)

        # Compute cosine similarity between the query vector and document vectors
        similarities = cosine_similarity(query_vector, document_vectors)

        # Get indices of documents or items sorted by similarity scores (descending order)
        top_indices = np.argsort(similarities[0])[::-1][:3]
    
    final_df=pd.read_csv('zomato_restaurant_final.csv')
    df=final_df[final_df['area'].str.contains(str(user_location).lower())]
    df1=df[df['cuisine'].str.contains(str(user_cuisine).lower())]
    if df1.empty:
        st.header("We could not find any matches for your input. However, here are some recommendations for you:")
        for idx in top_indices:
            st.title(final_df.loc[idx, 'names'])
            st.subheader("Details:")
            st.write("Rating:", final_df.loc[idx, 'ratings'])
            st.write("Cuisine:", final_df.loc[idx, 'cuisine_list'])
            st.write("Location:", final_df.loc[idx, 'area_list'])
            st.write("Price:", final_df.loc[idx, 'price for two'])
        
    else:
        # Display top recommended recipes
        st.header("\nTop Recommended Restaurants:")
        for idx in top_indices:
            st.title(final_df.loc[idx, 'names'])
            st.subheader("Details:")
            st.write("Rating:", final_df.loc[idx, 'ratings'])
            st.write("Cuisine:", final_df.loc[idx, 'cuisine_list'])
            st.write("Location:", final_df.loc[idx, 'area_list'])
            st.write("Price:", final_df.loc[idx, 'price for two'])
            # st.subheader("Restaurant Name: ")
            # st.write(final_df.loc[idx, 'names'])
            # st.subheader("Restaurant Rating: ")
            # st.write(final_df.loc[idx, 'ratings'])
            # st.subheader("Cuisine: ")
            # st.write(final_df.loc[idx, 'cuisine_list'])
            # st.subheader("Location: ")
            # st.write(final_df.loc[idx, 'area_list'])
            # st.subheader("Price: ")
            # st.write(final_df.loc[idx, 'price for two'])
        
    