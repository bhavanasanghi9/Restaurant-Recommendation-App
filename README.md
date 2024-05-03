# Restaurant-Recommendation-App
A streamlit app used to generate recommendations for restaurants in bangalore

## There are two versions of the app
### The first version recommends restaurants based on 2 inputs - choice of location, choice of cuisine

version 1 - https://github.com/bhavanasanghi9/Restaurant-Recommendation-App/blob/main/restaurant-recommendation-model.ipynb

This notebook contains - 
1. Code to scrape restaurant details from zomato
2. Cleaning and data pre-processing
3. Using TF-IDF vectorization and cosine similarity, recommedations are made 
link to streamlit app code - https://github.com/bhavanasanghi9/Restaurant-Recommendation-App/blob/main/restaurant_recommend_app.py

version 2 - https://github.com/bhavanasanghi9/Restaurant-Recommendation-App/blob/main/restaurant-recommendation-model-updated.ipynb

This notebook contains - 
1. Code to scrape restaurant details from zomato
2. Cleaning and data pre-processing
3. Using Word2Vec model and cosine similarity, recommedations are made 
link to streamlit app code - https://github.com/bhavanasanghi9/Restaurant-Recommendation-App/blob/main/restaurant_recommend_app.py

Link to app - https://restaurant-recommendation-app-yrbjfi5fcoprvjbbxvpjuu.streamlit.app/

### The second version recommends restaurants based on 3 inputs - choice of location, choice of cuisine, user preference

link to notebook - https://github.com/bhavanasanghi9/Restaurant-Recommendation-App/blob/main/restaurant_rec_model_with_reviews.ipynb

This notebook contains - 
1. Code to scrape restaurant reviews for all bangalore restaurants from zomato
2. Cleaning and data pre-processing of reviews data
3. Using TF-IDF vectorization and cosine similarity, recommendations are made
Link to streamlit app code - https://github.com/bhavanasanghi9/Restaurant-Recommendation-App/blob/main/recommender_pref.py
Link to app - https://restaurant-recommendation-app-v2-xqwywdbnyyjv5u6psarrcq.streamlit.app/


