import streamlit as st
import pandas as pd
import numpy as np
pip install scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression


s = pd.read_csv("social_media_usage.csv")

def clean_sm(x):     
    x=np.where(x == 1, 1, 0)
    return x


ss = pd.DataFrame({
    "sm_li":clean_sm(s["web1h"]),
    "income":np.where(s["income"]<=9,s["income"],np.nan),
    "education":np.where(s["educ2"]<=8,s["educ2"],np.nan),
    "parent":clean_sm(s["par"]),
    "married":clean_sm(s["marital"]),
    "female":np.where(s["gender"]==2,1,0),
    "age":np.where(s["age"]<=97,s["age"],np.nan)
})

ss = ss.dropna()

y = ss["sm_li"]
X = ss[["income","education","parent","married","female","age"]]

X_train, X_test, y_train, y_test = train_test_split(X,
                                                   y,
                                                   stratify=y, #same proportion of target in training and test set
                                                   test_size=0.2, #hold out 20% of data for testing
                                                   random_state=987) #set for reproducibility

lr = LogisticRegression()

lr.fit(X_train, y_train)


st.write("""
# LinkedIn User Prediction App

This app predicts the probability an individual is a **LinkedIn user!**
""")

# getting input data from the user

st.write("""Please fill in the below information to determine the probability you are a LinkedIn user.
""")

st.write("""
Household Income Selections: 
""")
df_income = pd.DataFrame({'selection':[1,2,3,4,5,6,7,8,9,98,99],'amount':["Less than 10,000","10 to under 20,000","20 to under 30,000","30 to under 40,000","40 to under 50,000","50 to under 75,000","75 to under 100,000","100 to under 150,000","150 or more","Don't know","Refuse"]})
st.write(df_income)

income = st.selectbox("Income",
                options = ["1",
                           "2",
                           "3",
                           "4",
                           "5",
                           "6",
                           "7",
                           "8",
                           "9",
                           "98",
                           "99"])

#st.write(f"Income:{income}")

st.write("""Highest Level of School/Degree Completed Selections:
""")
df_education = pd.DataFrame({'selection':[1,2,3,4,5,6,7,8,98,99],'level':["Less than high school","High school incomplete","High school graduate","Some college, no degree","Two-year associate degree","Four-year college","Some postgraduate, no degree","Postgraduate degree","Don't know","Refuse"]})
st.write(df_education)

education = st.selectbox("Education",
                options = ["1",
                           "2",
                           "3",
                           "4",
                           "5",
                           "6",
                           "7",
                           "8",
                           "98",
                           "99"])

st.write("""Parent of a Child Under 18 Living in Your Home Selections:
""")
df_parent = pd.DataFrame({'selection':[1,2,8,9],'options':["Yes","No","Don't know","Refuse"]})
st.write(df_parent)

parent = st.selectbox("Parent",
                options = ["1",
                           "2",
                           "8",
                           "9"])

#st.write(f"Parent:{parent}")

st.write("""Current Marital Status Selections:
""")
df_married = pd.DataFrame({'selection':[1,2,3,4,5,6,8,9],'options':["Married","Living with a partner","Divorced","Separated","Widowed","Never been married","Don't know","Refuse"]})
st.write(df_married)

married = st.selectbox("Married",
                options = ["1",
                           "2",
                           "3",
                           "4",
                           "5",
                           "6",
                           "8",
                           "9"])

#st.write(f"Married:{married}")

st.write("""Gender Selections:
""")
df_gender = pd.DataFrame({'selection':[1,2,3,98,99],'options':["male","female","other","Don't know","Refuse"]})
st.write(df_gender)
female = st.selectbox("Gender",
                options = ["1",
                           "2",
                           "3",
                           "98", 
                           "99"])

#st.write(f"Gender:{female}")

age = st.slider(label="Age")

#st.write(f"Age:{age}")

person = [age,education,parent,married,female,age]

probs = lr.predict_proba([person])

st.write(f"Propability this person is a LinkedIn user: {probs[0][1]}")

#st.button('Probability of being a LinkedIn User' on_click=predict)
