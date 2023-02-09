import streamlit as st
import pandas as pd
import numpy as np
import pickle

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import VotingClassifier

st.write("""
# Cognitive Prediction Application
""")

st.sidebar.header('User Input Features')

st.sidebar.markdown("""
[Example CSV input file](https://www.kaggle.com/datasets/mstiquecida/cleaned-dataset-for-alzheimer-disease-predication)
""")

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        pt_age = st.sidebar.slider("Age", 0, 100, 81) 
        pt_gender = st.sidebar.selectbox("Gender", ("Male","Female")) 
        pt_status = st.sidebar.selectbox("Marital_Status",("Married", "Widowed", "Divorced", "Never married", "Unknown")) 
        APOE4  = st.sidebar.slider("APOE4", 1, 4, 1)
        CDRSB  = st.sidebar.slider("CDRSB", 0.0, 20.0, 4.5)
        ADAS11  = st.sidebar.slider("ADAS11", 0.0, 100.0, 22.0)
        ADAS13  = st.sidebar.slider("ADAS13", 0.0, 100.0, 31.0)
        MMSE    = st.sidebar.slider("MMSE", 0, 50, 20)
        RAVLT_Immediate = st.sidebar.slider("RAVLT_Immediate", 0, 100, 22)
        Ventricles   = st.sidebar.slider("Ventricles", 5650, 200000, 84599)
        Hippocampus    = st.sidebar.slider("Hippocampus", 2219, 11207, 5319)
        Whole_Brain   = st.sidebar.slider("Whole_Brain", 649091, 1500000, 1129830)
        Entorhinal    = st.sidebar.slider("Entorhinal", 1050, 6715, 1791)
        Mid_Temp   = st.sidebar.slider("Mid_Temp", 8044, 32500, 18422)
        data = {"pt_age":pt_age,
                "pt_gender":pt_gender,
                "pt_status":pt_status,
                "APOE4": APOE4,
                "CDRSB": CDRSB,
                "ADAS11": ADAS11,
                "ADAS13": ADAS13,
                "MMSE": MMSE,
                "RAVLT_Immediate": RAVLT_Immediate,
                "Ventricles": Ventricles,
                "Hippocampus": Hippocampus,
                "Whole_Brain": Whole_Brain,
                "Entorhinal": Entorhinal,
                "Mid_Temp":Mid_Temp
                }
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

# Combines user input features with the entire dataset
# This will be useful for the encoding phase
health_status_raw = pd.read_csv('CleanHealthData.csv')
health_status = health_status_raw.drop(columns=['DX'])
df = pd.concat([input_df,health_status],axis=0)

# Encoding of ordinal features
encode = ['pt_gender','pt_status']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]
df = df[:5] # Selects only the first row (the user input data)

# Displays the user input features
st.subheader('User Input features')

if uploaded_file is not None:
    st.write(df)
else:
    st.write('Adjust readings accordingly, or upload a CSV file. Currently using default data parameters (shown below).')
    st.write(df)

# Reads in saved classification model
load_clf = pickle.load(open('Adb_model_clf.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)


st.subheader('Predicted Cognitive Stage')
cognitive_stage = np.array(['CN - Cognitive Normal','SMC - Significant Memory Concern','EMCI - Early Cognitive Impairment','LMCI - Late Cognitive Impairment','AD - Alzheimer Disease'])
st.write(cognitive_stage[prediction])

st.subheader('Cognitive Stage Probability')
st.write(prediction_proba)
