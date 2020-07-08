import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


df = pd.read_csv("bank.csv")



st.title('Dataframe basic informations')
st.sidebar.subheader('Basica exploratory analysis options')
if st.sidebar.checkbox('Basic informations'):
    if st.sidebar.checkbox('Head'):
        st.subheader('Dataframe head:')
        st.write(df.head())
    elif st.sidebar.checkbox('Tail'):
        st.subheader('Dataframe Tail:')
        st.write(df.tail())
    if st.sidebar.checkbox('Describe'):
        st.subheader('Data Descripition')
        st.write(df.describe())
    if st.sidebar.checkbox('IsNull'):
        st.subheader('Missing values')
        st.write(df.isnull().sum())
        
        
st.title('Dataframe plots')
st.sidebar.subheader('Data visualization options')
if st.sidebar.checkbox('Graphics'):
    if st.sidebar.checkbox('Count Plot'):
        st.subheader('Count Plot')
        column_count_plot = st.sidebar.selectbox("Choose a column to plot count",df.columns)
        hue_opt = st.sidebar.selectbox("Optional categorical variables (countplot hue)",df.columns.insert(0,None))
        if st.checkbox('Plot Countplot'):
            fig = sns.countplot(x=column_count_plot,data=df,hue=hue_opt)
            st.pyplot()
            
            
    if st.sidebar.checkbox('Histogram'):
        st.subheader('Distplot')
        if st.checkbox('Dist plot'):
            column_dist_plot = st.sidebar.selectbox("Optional categorical variables (countplot hue)",df.columns)
            fig = sns.distplot(df[column_dist_plot])
            st.pyplot()
            
            
    if st.sidebar.checkbox('Heatmap'):
        st.subheader('HeatMap')
        fig = sns.heatmap(df.corr(),annot=True, annot_kws={"size": 7}, linewidths=.5)
        st.pyplot()
        
        
    if st.sidebar.checkbox('Boxplot'):
        st.subheader('Boxplot')
        column_box_plot_X = st.sidebar.selectbox("X (Choose a column):",df.columns.insert(0,None))
        column_box_plot_Y = st.sidebar.selectbox("Y (Choose a column - only numerical):",df.columns)
        hue_box_opt = st.sidebar.selectbox("Optional categorical variables (boxplot hue)",df.columns.insert(0,None))
        if st.checkbox('Plot Boxplot'):
            fig = sns.boxplot(x=column_box_plot_X, y=column_box_plot_Y,data=df,palette="Set3")
            st.pyplot()
    if st.sidebar.checkbox('Pairplot'):
        st.subheader('Pairplot')
        hue_pp_opt = st.sidebar.selectbox("Optional categorical variables (pairplot hue)",df.columns.insert(0,None))
        st.info("This action may take a while.")
        fig = sns.pairplot(df,palette="coolwarm")
        st.pyplot()



pickle_in = open("classifier.pkl","rb")
classifier=pickle.load(pickle_in)



def predict_note_authentication(Duration,Age,Day,Balance):
    
    
   
    prediction=classifier.predict([[Duration,Age,Day,Balance]])
    print(prediction)
    return prediction



def main():
    st.title("Bank Deposit")
    #html_temp = """
    #<div style="background-color:tomato;padding:10px">
    #<h2 style="color:white;text-align:center;">Streamlit Bank Deposit ML App </h2>
    #</div>
    #"""
    #st.markdown(html_temp,unsafe_allow_html=True)
    Duration = st.text_input("Duration"," ")
    Age = st.text_input("Age"," ")
    Day = st.text_input("Day"," ")
    Balance = st.text_input("Balance"," ")
    result=""
    if st.button("Predict"):
        result=predict_note_authentication(Duration,Age,Day,Balance)
    st.success('The output is {}'.format(result))
  

if __name__=='__main__':
    main()
        


        
