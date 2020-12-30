import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import plotly
import seaborn as sns
from scipy import stats

# plt.style.use(["dark_background"])

from sklearn.svm import SVC
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from category_encoders import LeaveOneOutEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from functions_pkg import print_vif, predictions_df
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
)
from sklearn.metrics import precision_score, recall_score
import streamlit as st



# def main():
st.title("Cardiovascular Disease Patient Classifier")
st.markdown("Suffering from cardiovascular disease?")


path = "cardiovascular_disease_prediction/cardio_train.csv"
raw_df = pd.read_csv(path, sep=";", index_col="id")



    
# checkbox for loading data
@st.cache
def load_data():
    # dataset file info
    data = pd.read_csv(path, sep=";", index_col="id")

    # new column name mapping
    mapping = {
        "ap_hi": "bp_hi",
        "ap_lo": "bp_lo",
        "gluc": "glucose",
        "alco": "alcohol",
        "cardio": "disease",
    }
    # column renaming
    data = data.rename(columns=mapping)

    # change gender to 0-1 binary
    data.loc[:, "gender"] = data.gender - 1

    # reduce interval in cholesterol & glucose from 1-3 to 0-2
    data.loc[:, "cholesterol"] = data.cholesterol - 1
    data.loc[:, "glucose"] = data.glucose - 1

    # blood pressure difference column
    data["bp_diff"] = data.bp_hi - data.bp_lo

    # BMI column to replace height and weight
    # bmi = weight (kgs) / (height (m))^2
    data["bmi"] = data.weight / (data.height / 100) ** 2

    # drop height and weight columns from df
#     data = data.drop(columns=["height", "weight"])
    
    # return cleaned dataset
    return data



def split(df):# Split the data to 'train and test' sets
        req_cols = ['age', 'gender', 'bp_hi', 'bp_lo', 'cholesterol', 'glucose', 'smoke', 'alcohol', 'active', 'bp_diff', 'bmi']
        X = df[req_cols] # Features for our algorithm
        y = df.disease
        X = df.drop(columns=['disease'])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=28)
        return X_train, X_test, y_train, y_test

def plot_metrics(metrics_list):
    if 'Confusion Matrix' in metrics_list:
        st.subheader("Confusion Matrix")
        plot_confusion_matrix(model, X_test, y_test, display_labels=class_names)
        st.pyplot()

    if 'ROC Curve' in metrics_list:
        st.subheader("ROC Curve")
        plot_roc_curve(model, X_test, y_test)
        st.pyplot()

    if 'Precision-Recall Curve' in metrics_list:
        st.subheader('Precision-Recall Curve')
        plot_precision_recall_curve(model, X_test, y_test)
        st.pyplot() 

df = load_data()

    
st.sidebar.title("Model")
#st.sidebar.markdown("Choose model and parameters")

class_names = ['Disease', 'No Disease']
x_train, x_test, y_train, y_test = split(df)

option = st.sidebar.selectbox("Model Option", ("Data", "Sample", "Model"))


if option == "Data":
    
    # data information
    st.subheader("Types of input features:")
    st.write("   - Objective: factual information")
    st.write("   - Examination: results of medical examination")
    st.write("   - Subjective: information given by the patient")
    
    if st.sidebar.checkbox("raw data", False):
        st.subheader("Raw Dataset")
        st.write(raw_df.head()) 
        
#     if st.sidebar.checkbox("clean data", False):
    st.subheader("Clean Data")
    st.write(df.head())
    st.subheader("Variables")
    st.write("1. age (days)")
    st.write("2. gender (0=female|1=male)")
    st.write("3. height (cm)")
    st.write("4. weight (kg)")
    st.write("5. bp_hi [systolic blood pressure]")
    st.write("6. bp_lo [diastolic blood pressure]")
    st.write("7. cholesterol [normal (0) | high (1) | very high(2)]")
    st.write("8. glucose [normal(0) | high(1) | very high(2)]")
    st.write("9. smoke \[smoking?] (0=no|1=yes)")
    st.write("10. alcohol \[drinking?] (0=no|1=yes)")
    st.write("11. active \[physically active?] (0=no|1=yes)")
    st.write("12. disease [presence (1) or absence (0) of cardiovascular disease]")
    st.write("13. bp_diff [bp_hi - bp_lo]")
    st.write("14. bmi [body mass index]")

num_cols = ["bp_lo", "bp_hi", "bp_diff", "bmi", "height", "weight"]

if option == "Sample":
    st.subheader("By the numbers:")
    stat = st.sidebar.selectbox("Variable", ("gender", "cholesterol", "glucose", "smoke", "alcohol", "active", "bp_lo", "bp_hi", "bp_diff", "bmi", "height", "weight", "disease"))
    feat = df[stat]
    
    # plotting
    fig, ax = plt.subplots()
    
    # numeric vars
    if stat in num_cols:
        sns.distplot(feat)
        
        st.pyplot(fig)
    
    # categorical vars
    else:
        vals = feat.value_counts()
        st.write(vals)
        sns.countplot(vals)
        # plot
        st.pyplot(fig)
    
    
    
    

        
    













