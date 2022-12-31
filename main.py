import streamlit as st
import pandas as pd
import numpy as np
from preprocess import Preprocess
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, StandardScaler

# Store the initial value of widgets in session state
st.write("# Web Madenciliği Projesi")
st.write("    ## Başlamadan önce lütfen aşağıdaki bilgileri doldurunuz.")

st.write("### Veri seti yükleyiniz.")
uploaded_file = st.file_uploader("Veri setini yükleyiniz.", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file
    )   
    st.write("Veri seti başarıyla yüklendi.")
    st.write(data.head())

# create a button to start preprocessing

col1, col2 = st.columns(2, gap='medium')
# select whether to se a 2D or 3D conformation
with col1:
    encoder_format = st.selectbox("Encoder seçiniz:", ['One Hot Encoder', 'Label Encoder'],           
                                help='Encoder türünü belirleyip veriyi ona göre encode ediniz.')
# select the download format
with col2:
    scaler_format = st.selectbox('Scaler seçiniz:', ['MinMaxScaler','StandartScaler'], key='',
                                help="""Scaler türünü belirleyip veriyi ona göre scale ediniz.""")

if encoder_format == 'One Hot Encoder':
    encoder = OneHotEncoder()
elif encoder_format == 'Label Encoder':
    encoder = LabelEncoder()

if scaler_format == 'MinMaxScaler':
    scaler = MinMaxScaler()
elif scaler_format == 'StandartScaler':
    scaler = StandardScaler()

data_prep_col1,data_prep_col2 = st.columns(2)


importance_threshold = st.slider('Importance Threshold', min_value=0.0, max_value=.1, value=0.05, step=0.005)
corr_threshold = st.slider('Correlation Threshold', min_value=0.0, max_value=1.0, value=0.5, step=0.1)

with data_prep_col1:
    if st.button('Veri Önişlemeyi Başlat'):
        try:
            print("Encoder coming from selection is",encoder)
            preprocess = Preprocess(data,encoder,scaler,importance_threshold,corr_threshold)
            data = preprocess.preprocess()
            st.write(data.head())
        except:
            st.write("Lütfen önce veri setini yükleyiniz.")
            

with data_prep_col2:
    if st.button('Veri Önişlemeyi Defaul Değerlerle Başlat'):
        try:
            preprocess = Preprocess(data,LabelEncoder(),MinMaxScaler(),0.01,0.8)
            data = preprocess.preprocess()
            st.write(data.head())
        except:
            st.write("Lütfen önce veri setini yükleyiniz.")


