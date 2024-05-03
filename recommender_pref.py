import pickle
import streamlit as st
import pandas as pd 
import numpy as np
import re
from sklearn.preprocessing import normalize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("Let's help you decide where to eat today!")

#Get user to select the desired location

location_list=['I am willing to go anywhere today','Indiranagar', 'Koramangala', 'St. Marks Road', 'Cunningham Road',
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
location_mapping = {word: word for word in location_list}
location_mapping['I am willing to go anywhere today']=''

user_location = st.selectbox(
    'Which area would you like to visit today',
    tuple(location_list))

st.write('You selected:', user_location)
user_location=location_mapping.get(user_location)

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

preference={'I am not feeling picky today':'',
    'romantic ambience':'romantic',
            'cozy ambience':'cozy',
            'amazing ambience':'ambience',
            'date night':'date',
            'live music':'live',
            'good service':'service',
            'quick service':'quick',
            'amazing dj':'dj',
            'party place':'dance',
            'breakfast vibe':'breakfast',
            'family friendly':'family',
            'kid friendly':'kid',
            'affordable food':'affordable',
            'elaborate menu':'elaborate',
            'known for cocktails':'cocktails',
            'pub vibes':'pub',
            'pet friendly':'pet',
            'brunch vibes':'brunch',
            'value for money':'value money',
            'generous food portion':'portion',
            
            }
preference_list = list(preference.keys())

# Allow user to select up to 2 options
selected_options = st.multiselect('Select up to 2 preferences (lets not be too picky):', preference_list, max_selections=3)
# Display the selected options
st.write('Your selected preferences:', selected_options)



def input_transformer(location, cuisine,selected_options):
    location=str(location)
    cuisine=str(cuisine)
    location=location.replace(' ','')
    if not selected_options:
        user_input=location + ' ' + cuisine
    else:
        selected_options[0]=preference.get(selected_options[0])
        if len(selected_options)==2:
            selected_options[1]=preference.get(selected_options[1])
        preferences=' '.join(selected_options)
        user_input = location + ' ' + cuisine+ ' '+preferences
    final_output=('').join(user_input.lower().split('  '))
    return final_output

preprocessed_input = input_transformer(user_location, user_cuisine, selected_options)
query_tokens=preprocessed_input.split()

#st.button("Reset", type="primary")
if st.button('Generate Recommendations'):
    if not selected_options:
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
            st.header("We could not find any matches in your area for your input. However, here are some recommendations for you:")
            for idx in top_indices:
                st.title(final_df.loc[idx, 'names'])
                st.subheader("Details:")
                st.write("Rating:", final_df.loc[idx, 'ratings'])
                st.write("Cuisine:", final_df.loc[idx, 'cuisine_list'])
                st.write("Location:", final_df.loc[idx, 'area_list'])
                st.write("Price:", final_df.loc[idx, 'price for two'])
            
        else:
            # Display top recommended recipes
            st.header("\nTop Recommendaed Restaurants:")
            for idx in top_indices:
                st.title(final_df.loc[idx, 'names'])
                st.subheader("Details:")
                st.write("Rating:", final_df.loc[idx, 'ratings'])
                st.write("Cuisine:", final_df.loc[idx, 'cuisine_list'])
                st.write("Location:", final_df.loc[idx, 'area_list'])
                st.write("Price:", final_df.loc[idx, 'price for two'])
    else:
        
        review_df=pd.read_csv('final_zomato_rests_with_reviews.csv')
        data1=review_df[review_df['area'].str.contains(str(user_location).lower())]
        data2=data1[data1['cuisine'].str.contains(str(user_cuisine).lower())]
        if data2.empty:
            st.header(f'\nWe could not find matches in your area for your input. However, here are some {user_cuisine} recommendations for you')
            data2=review_df[review_df['cuisine'].str.contains(str(user_cuisine).lower())]
        data2.reset_index(drop=True,inplace=True)
        
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(data2['area'] + ' ' + data2['cuisine'] + ' ' + data2['final_reviews'])

        # Convert to DataFrame for display
        tfidf_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

        # Normalize TF-IDF matrix
        X_normalized = normalize(X, norm='l2', axis=1)
        
        user_input_vector = vectorizer.transform([preprocessed_input])

        # Calculate cosine similarity between user input and recipes
        similarity_scores = cosine_similarity(user_input_vector, X_normalized)

        # Get indices of top recommended recipes (top 3 in this example)
        top_indices = similarity_scores.argsort()[0][-3:][::-1]
    
        dummy=data2[data2['final_reviews'].str.contains(str(selected_options[0]).lower())]
        if len(selected_options)==2:
            dummy1=dummy[dummy['final_reviews'].str.contains(str(selected_options[1]).lower())]
        else:
            dummy1=dummy
        if dummy1.empty:
            # Display top recommended recipes
            st.header("\nWe could not find restaurants catering to your exact preferences. However, we think you will love:")
            for idx in top_indices:
                st.title(data2.loc[idx, 'names'])
                st.subheader("Details:")
                st.write("Rating:", data2.loc[idx, 'ratings'])
                st.write("Cuisine:", data2.loc[idx, 'cuisine_list'])
                st.write("Location:", data2.loc[idx, 'area_list'])
                st.write("Price:", data2.loc[idx, 'price for two'])
        else:
        # Display top recommended recipes
            st.header("\nTop Recommended Restaurants:")
            for idx in top_indices:
                st.title(data2.loc[idx, 'names'])
                st.subheader("Details:")
                st.write("Rating:", data2.loc[idx, 'ratings'])
                st.write("Cuisine:", data2.loc[idx, 'cuisine_list'])
                st.write("Location:", data2.loc[idx, 'area_list'])
                st.write("Price:", data2.loc[idx, 'price for two'])
            
        
        

        
        
        
        
        
    