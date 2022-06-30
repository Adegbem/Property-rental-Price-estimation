"""
documentation
"""
import base64
import os.path
import re
import time
import joblib
import os

# Data dependencies
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Streamlit dependencies
import streamlit as st
from PIL import Image

# Model_map
model_map = {'Ridge': 'Ridge_model.pkl', 'GradientBoostingRegressor': 'XGboost_model.pkl'}


# Normalize data
def normalize(predictors: dict):
    df = pd.read_csv('normalize_df.csv')
    df.drop('Unnamed: 0', axis=1, inplace=True)
    temp_df = pd.DataFrame(predictors, index=[0])
    scaler = MinMaxScaler()
    scaler.fit_transform(df)
    X_scaled = scaler.transform(temp_df)
    X_normalize = pd.DataFrame(X_scaled, columns=predictors.keys())
    return X_normalize


# creating a sentiment_map and other variables
palette_color = sns.color_palette('dark')

# """### gif from local file"""
# file_ = open("thank_you.gif", "rb")
# contents = file_.read()
# data_url = base64.b64encode(contents).decode("utf-8")
# thumps_down = Image.open("thums_down.webp")
# thumps_up = Image.open("thumps_up.webp")
# nuetral = Image.open("neutral.webp")
screaming_emoji = Image.open("screaming.jpg")


#
# file_.close()


# The main function where we will build the actual app
def main():
    """Property rentals price estimation App with Streamlit """
    st.set_page_config(
        page_title="Property rentals pricing App",
        page_icon="🧊",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://www.linkedin.com/in/adegbem/',
            'Report a bug': "https://www.linkedin.com/in/adegbem/",
            'About': "# This is a header. This is an *Property rentals* pricing app!"
        }
    )
    # st.set_page_config(page_title="Property rentals pricing", page_icon=":hash:", layout="centered")

    # Creating sidebar with selection box -
    # you can create multiple pages this way
    menu = ["Landing Page", "Price Prediction", "EDA", "About the APP"]
    selection = st.sidebar.selectbox("Choose Option", menu)
    ss = None
    if selection == "Landing Page":
        col1, col2, col3 = st.columns(3)
        with col2:
            st.subheader("Property Rental Pricing App")

    elif selection == "Price Prediction":
        st.subheader("Price Prediction")
    elif selection == "EDA":
        st.subheader("Exploration of Property Features")
    else:
        st.header("App Documentation")

    # Landing page
    landing = Image.open("house-rent_logo.jpg")
    if selection == "Landing Page":
        st.image(landing)  # , height=1500)
        time.sleep(3)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Property rentals pricing:")
        with col2:
            st.markdown(
                'This app is so cool, it let you know how much you can earn per night of renting out your **living space**.\n'
                'It also give insight into what criteria are being used to estimate this price per each property')

        # bu1 = st.button("Next page")
        #
        # if bu1:
        #     ss = st.selectbox("Choose Option", ["Price Prediction"])

    # Text Prediction page
    if selection == "Price Prediction" or ss == "Price Prediction":
        st.markdown("Select corresponding features of your **Living Space**")

        uploaded_file = st.file_uploader("Upload a picture of your Living Space")

        # Creating a parameters for user input
        col1, col2 = st.columns(2)

        with col1:
            latitude = st.slider("Latitude", 37.70463, 37.82879, 37.76671)

        with col2:
            minimum_nights = st.slider("Minimum Night", 1, 365, 30)

        col3, col4 = st.columns(2)

        with col3:
            bathrooms = st.slider("No. of Bathrooms", 0, 10, 5)

        with col4:
            bedrooms = st.slider("No. of Bedrooms", 0, 5, 3)

        # longitude = st.slider("Longitude", -122.440815, -122.36857, -122.51306)

        col5, col6 = st.columns(2)

        with col5:
            property_type = st.selectbox("Property type",
                                         ['Apartment', 'House', 'Condominium', 'Guest suite', 'Boutique hotel',
                                          'Hotel', 'Townhouse', 'Serviced apartment', 'Hostel', 'Loft',
                                          'Bed and breakfast', 'Guesthouse', 'Aparthotel', 'Other', 'Bungalow',
                                          'Resort', 'Cottage', 'Villa', 'Castle', 'Cabin', 'Tiny house',
                                          'Earth house', 'In-law', 'Camper/RV', 'Dome house', 'Hut'])
        with col6:
            room_type = st.selectbox("Room type", ['Entire home/apt', 'Private room', 'Shared room', 'Hotel room'])

        df_encoded = ['latitude', 'bathrooms', 'bedrooms', 'minimum_nights',
                      'property_type_Apartment', 'property_type_Bed and breakfast',
                      'property_type_Boutique hotel', 'property_type_Bungalow',
                      'property_type_Cabin', 'property_type_Camper/RV',
                      'property_type_Castle', 'property_type_Condominium',
                      'property_type_Cottage', 'property_type_Dome house',
                      'property_type_Earth house', 'property_type_Guest suite',
                      'property_type_Guesthouse', 'property_type_Hostel',
                      'property_type_Hotel', 'property_type_House', 'property_type_Hut',
                      'property_type_In-law', 'property_type_Loft', 'property_type_Other',
                      'property_type_Resort', 'property_type_Serviced apartment',
                      'property_type_Tiny house', 'property_type_Townhouse',
                      'property_type_Villa', 'room_type_Hotel room', 'room_type_Private room',
                      'room_type_Shared room', 'price']

        dicting = {'latitude': latitude, 'bathrooms': bathrooms,
                   'bedrooms': bedrooms, 'minimum_nights': minimum_nights}

        st.info("Prediction with ML Models")
        for x in df_encoded:
            if re.match(r'property_type', x):
                matching = re.search(r'[a-zA-Z ]+', x.split('_')[-1])
                if matching.group() == property_type:
                    dicting[x] = 1
                else:
                    dicting[x] = 0

        for x in df_encoded:
            if re.match(r'room_type', x):
                matching = re.search(r'[a-zA-Z ]+', x.split('_')[-1])
                if matching.group() == room_type:
                    dicting[x] = 1
                else:
                    dicting[x] = 0

        model_name = st.selectbox("Choose Model", model_map.keys())
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            bu = st.button("Estimator Calculator")
        if bu:
            # Load your .pkl file with the model of your choice + make predictions
            # Try loading in multiple models to give the user a choice
            predictor = joblib.load(open(os.path.join(model_map[model_name]), "rb"))
            df_param = normalize(dicting)
            prediction = predictor.predict(df_param)

            # When model has successfully run, will print prediction
            # You can use a dictionary or similar structure to make this output
            # more human interpretable.
            col1, col2, col3 = st.columns(3)
            with col2:
                if uploaded_file is not None:
                    # To read file as bytes:
                    st.image(uploaded_file)

            col1, col2 = st.columns(2)
            with col1:
                if model_name == 'GradientBoostingRegressor':
                    st.write(f"The estimated price for your living space is about {round(prediction[0], 2)} USD")
                else:
                    st.write(f"The estimated price for your living space is about {round(prediction[0][0], 2)} USD")
            with col2:
                st.markdown(
                    "This is just a rough estimate, who knows you might get to earn more if you let us help you rent out "
                    "your space, What are you waiting for!!!")
            st.image(screaming_emoji)
            st.balloons()

    # Building out the "Information" page
    # if selection == "Information":
    #     st.info("Brief Description")

    #     st.markdown(" ")

    #     # hash_pick = st.checkbox('Hash-Tag')
    #     # if hash_pick:
    #     #     val = st.selectbox("Choose Tag type", ['Hash-Tag', 'Mentions'])
    #         # sentiment_select = st.selectbox("Choose Option", sentiment_map)
    #         # iter_hash_select = st.slider('How many hash-tag', 1, 20, 10)

    #         # if val == 'Hash-Tag':
    #         #     st.info("Popular Hast Tags")
    #         # else:
    #         #     st.info("Popular Mentions")
    #         # valc = 'hash_tag' if val == 'Hash-Tag' else 'mentions'
    #         # result = tags(sentiment_cat=sentiment_map[sentiment_select], iter_hash_num=iter_hash_select,
    #         #               col_type=valc)

    #     # result = tags(sentiment_cat=sentiment_map[sentiment_select], iter_hash_num=iter_hash_select,
    #     #                   col_type=valc)

    #     col1, col2 = st.columns(2)
    #     col1.success('1')
    #     col2.success('Important/most used words')

    #     with col1:
    #         sentiment_select = st.selectbox("Choose Option", sentiment_map)
    #         iter_hash_select = st.slider('How many hash-tag', 1, 20, 10)

    #     with col2:
    #         st.write('The word cloud function goes in here')

    #         st.markdown(" ")

    #     col3, col4 = st.columns(2)
    #     col3.success('Popular hashtags')
    #     col4.success('mentions')

    #     with col3:
    #         source = pd.DataFrame({
    #             'Frequency': result.values(),
    #             'Hash-Tag': result.keys()})
    #         st.write("List of popular hashtags function associated with sentiment goes in here")

    #     with col4:
    #         chart_data = pd.DataFrame(np.random.randn(50, 3), columns=["a", "b", "c"])

    #         st.bar_chart(chart_data)

    # Building out the prediction page
    # if selection == "Upload File":
    #     st.info("Prediction with ML Models")
    #
    #     data_file = st.file_uploader("Upload CSV", type=['csv'])
    #     if st.button("Process"):
    #         if data_file is not None:
    #             df = pd.read_csv(data_file)
    #             tweet_process = df['message'].apply(cleaning)
    #             # vectorizer = CountVectorizer(analyzer = "word", max_features = 8000)
    #             # reviews_vect = vectorizer.fit_transform(df['cleaned_tweet'])
    #             model_name = 'naive_bayes_model.pkl'
    #             predictor = joblib.load(open(os.path.join(model_name), "rb"))
    #             prediction = predictor.predict(tweet_process)
    #
    #             st.success(
    #                 pd.DataFrame(prediction).value_counts().plot(kind='bar'))
    #             plt.show()
    # st.pyplot(fig, use_container_width=True)

    # Creating a text box for user input
    # st.markdown('---')
    # tweet_text = st.text_area("Enter Text (Type in the box below)", " ")
    # st.markdown('---')
    # model_name = st.selectbox("Choose Model", model_map.keys())
    # tweet_process = cleaning(tweet_text)

    # st.write('You selected:', model_name)

    # if model_name == 'LogisticRegression':
    #     with st.expander("See explanation"):
    #         st.write("""Brief explanation of how the Logistic regression works goes in here""")

    # elif model_name == 'KNeighborsClassifier':
    #     with st.expander("See explanation"):
    #         st.write("""Brief explanation of how the KNN model works goes in here""")

    # elif model_name == 'SVC':
    #     with st.expander("See explanation"):
    #         st.write("""Brief explanation of how the SVC model works goes in here""")

    # elif model_name == 'DecisionTreeClassifier':
    #     with st.expander("See explanation"):
    #         st.write("""Brief explanation of how the Decision tree classifier model works goes in here""")

    # else:
    #     with st.expander("See explanation"):
    #         st.write("""Brief explanation of how the model works goes in here""")

    # st.markdown('---')

    #     if st.button("Classify"):
    #         # Load your .pkl file with the model of your choice + make predictions
    #         # Try loading in multiple models to give the user a choice
    #         predictor = joblib.load(open(os.path.join(model_map[model_name]), "rb"))
    #         prediction = predictor.predict([tweet_process])

    #         # When model has successfully run, will print prediction
    #         # You can use a dictionary or similar structure to make this output
    #         # more human interpretable.
    #         st.success("Text Categorized as: {}".format(prediction))
    #         if prediction == 1:
    #             st.write(""" **Thank you for supporting climate** 👈 """)
    #             st.markdown(f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
    #                         unsafe_allow_html=True)
    #     # # # st.markdown('---')
    #     # model_col,Accuracy_col=st.columns(2)
    #     # Accuracy_col.header('**Model Matrics**')

    # # Accuracy_col.subheader('mean absolute error')
    # # Accuracy_col.write(mean_absolute_error(y_test,prediction))
    # # Accuracy_col.subheader('mean square error')
    # # Accuracy_col.write(mean_squared_error(y_test,prediction))
    # Accuracy_col.subheader('R squared score error')
    # Accuracy_col.write(r2_score(y,prediction))

    # if selection == "EDA":
    #     hash_pick = st.checkbox('Hash-Tag')
    #     if hash_pick:
    #         val = st.selectbox("Choose Tag type", ['Hash-Tag', 'Mentions'])
    #         sentiment_select = st.selectbox("Choose Option", sentiment_map)
    #         iter_hash_select = st.slider('How many hash-tag', 1, 20, 10)
    #         if val == 'Hash-Tag':
    #             st.info("Popular Hast Tags")
    #         else:
    #             st.info("Popular Mentions")
    #         valc = 'hash_tag' if val == 'Hash-Tag' else 'mentions'
    #         result = tags(sentiment_cat=sentiment_map[sentiment_select], iter_hash_num=iter_hash_select,
    #                       col_type=valc)
    #         source = pd.DataFrame({
    #             'Frequency': result.values(),
    #             'Hash-Tag': result.keys()
    #         })
    #         val = np.array(list(result.values())).reshape(-1, 1)
    #         dd = (scaler.fit_transform(val)).reshape(1, -1)
    #         fig, ax = plt.subplots(1, 2, figsize=(10, 15))
    #         ax[0].pie(data=source, x=result.values(), labels=result.keys(), colors=palette_color)
    #         # explode=dd[0], autopct='%.0f%%')
    #         word_cloud = WordCloud(  # background_color='white',
    #             width=512,
    #             height=384).generate(' '.join(result.keys()))
    #         ax[1].imshow(word_cloud)
    #         ax[1].axis("off")
    #         plt.show()
    #         st.pyplot(fig, use_container_width=True)
    #
    #     word_pick = st.checkbox('Word Group(s)')
    #     if word_pick:
    #         st.info("Popular Group of Word(s)")
    #         sentiment_select_word = st.selectbox("Choose sentiment option", sentiment_map)
    #         word_amt = st.slider('Group of words', 1, 10, 5)
    #         group_amt = st.slider("Most frequent word groupings", 1, 10, 5)
    #         word_result = word_grouping(group_word_num=word_amt, ngram_iter_num=group_amt,
    #                                     sentiment_cat=sentiment_map[sentiment_select_word])
    #         st.table(pd.DataFrame({
    #             'Word group': word_result.keys(),
    #             'Frequency': word_result.values()
    #         }))


# Required to let Streamlit instantiate our web app.
if __name__ == '__main__':
    main()
