import streamlit as st
from Functions import load_data, headmap, normalize, rfml_pred, predict, plot_correlations_with_target
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from streamlit_option_menu import option_menu
def main():

    with st.sidebar:
        selected = option_menu(
            menu_title=None,
            options=["Home", "Heatmap", "Calculator"],
            icons=["house", "fire", "calculator"],
            menu_icon = "cast",
            default_index=0
        )

    if selected == "Home":
        st.title(f"{selected}")
        st.write("This app has the purpose to give a better understanding along with a possible estimation os Boston Medium house price and the most relevant variables")

        st.markdown("""
Here are some useful links:

- [Data](https://www.kaggle.com/datasets/altavish/boston-housing-dataset)
- [Orginal variables explanation](https://www.kaggle.com/code/prasadperera/the-boston-housing-dataset)
- [Main github repository](https://github.com/IJP99?tab=repositories)

Feel free to explore!
""")
    
    if selected == "Heatmap":
        st.title(f"{selected}")
        df = load_data()
        mens_headmap = """
The next graphs shows the importance between the different variables studied related to the final Mid-value of a house in Boston. 1 - Extremely important, 0 - compleatly irrelevant.
"""
        def stream_data(df):
            for word in mens_headmap.split(" "):
                yield word + " "
                time.sleep(0.02)

            fig = headmap(df)
            st.pyplot(fig)

        state=st.checkbox("Full heatmap", value=False)
        if state == True:
            st.write_stream(stream_data(df))

        target_column = st.selectbox("Select target column for correlation:", df.columns)
        
        if st.button("Show Correlation with Target Column"):
            fig = plot_correlations_with_target(df, target_column)
            st.pyplot(fig)
    
    if selected == "Calculator":
        st.title(f"{selected}")
        st.write("This calculator is intended to use a trained machine learning model to calculate the possible value of a house in Boston based on your preference in the down below variables")
        df = load_data()
        model = joblib.load("finalized_model.sav")
        scaler = joblib.load("scaler.sav")
    
        INDUS = st.slider("Select how important is for you to be near non-retail business.    0 = Not important, 30 =very important", 0,30)
    
        NOX = st.slider("Select how important is for you to be near of highway or power stations.    0 = I refuse to be near them, 1 = It doesn't bother me at all", 0.00,1.00)

        RM = st.select_slider("Number of rooms in the house", [1,2,3,4,5,6,7,8,9,10])

        DIS = st.slider("Weighted distances to five Boston employment centres. 0 = Close to those centers, 13 = Far from those centers", 0,13)

        TAX = st.number_input("full value real estate tax rate for $10,000 that you would be willing to bear", 100)

        PTRATIO = st.slider("Ratio of students per teacher in nearby schools", 10,25)

        LSTAT = st.slider("% lower status of the population", 0,45)

        if st.button("Get prediction"):
            prediction = predict(model, scaler, INDUS, NOX, RM, DIS, TAX, PTRATIO, LSTAT)
            predicted_value = prediction[0]
            st.write(f"The predicted house value is: {predicted_value * 10000:.2f}")

        st.write("**WARNING: Please consider this as a not real value and with the only purpose to give an approach based on your preference**")





if __name__ == '__main__':
    main()