import streamlit as st
import pandas as pd
import numpy as np

st.title('Decision Tree ECE 470')
st.write('Basic Decision Tree Created for an academic class and I decided to test out streamlit with it')

@st.cache
def load_data(nrows):
    data = pd.read_csv('la_cars_trimmed_features.csv', nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    return data

data_load_state = st.text('Loading data...')
data = load_data(10000)
data_load_state.text('Loading data... done!')

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)

st.subheader('Number of pickups by hour')
hist_values = np.histogram(data[price], bins=24, range=(0,18000))[0]
st.bar_chart(hist_values)

# Some number in the range 0-23
hour_to_filter = st.slider('hour', 0, 23, 17)
