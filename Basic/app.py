import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title('Data Analysis')
st.markdown('---')
st.write("Here are some of the ways on how you can do data analysis")

df = pd.read_csv('train.csv')

st.dataframe(df)

st.title('Data Types')
col1, col2 = st.columns([1,2])


with col1:
    # Filter the columns by datatypes
    st.markdown('### All about data types')
    dtypes_select = st.selectbox('Here are the columns for the dataset', df.dtypes.unique())
    hey = df.select_dtypes(include=dtypes_select, exclude=None)
    st.write(hey.columns)

with col2:
    # Plot data types
    fig = plt.figure(figsize=(12,7))
    dtypes_pie = plt.pie(df.dtypes.value_counts(), labels=df.dtypes.unique() ,autopct='%.0f%%')
    st.markdown('### Pie chart for each data types')
    st.pyplot(fig)

# Null Values
st.markdown('---')
st.title('Missing Values')

column_select = st.selectbox('Select a column to determine the missing values', df.columns)
st.write("Count of Null Values:", df[f'{column_select}'].isnull().sum())

fig2 = plt.figure(figsize=(12,7))
df.isnull().sum().plot.barh()
st.pyplot(fig2)

st.write("Use this slot for replacing null values")
st.markdown('---')

st.title('Data Visualization')

dv1, dv2, dv3 = st.columns(3)

with dv1:
        st.selectbox('Select the Datatype', df.dtypes.unique(), key = 123123)

with dv2:
    dsd = st.selectbox('Select the Column', df.dtypes.unique(), key = 223123)

with dv3:
    sds = st.selectbox('Select the chart type', df.dtypes.unique(), key = 333123)







# # plotting data on chart
# dtypes_pie = plt.pie(str(df.dtypes), autopct='%.0f%%')
