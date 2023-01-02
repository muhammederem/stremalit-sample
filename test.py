import streamlit as st
import pickle
import pandas as pd
import numpy as np
from preprocess import Preprocess
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, StandardScaler


# load the model from disk
st.write("### İstenilen modeli seçiniz.")
model = st.selectbox("Model seçiniz:", ['Random_Forest', 'SVC', 'XGBoost'],
                        help='Model türünü belirleyip veriyi ona göre eğitiniz.')
if model == 'Random_Forest':
    model = pickle.load(open('Random_Forest_model.pkl', 'rb'))
elif model == 'XGBoost':
    model = pickle.load(open('XGBoost_model.pkl', 'rb'))
elif model == 'SVC':
    model = pickle.load(open('SVC_model.pkl', 'rb'))

#get max value of each column


# tüm sütunlara uygun inputlar al
#age,gender,height_cm,weight_kg,body fat_%,diastolic,systolic,gripForce,sit and bend forward_cm,sit-ups counts,broad jump_cm,class
st.write("### Modelin tahmin etmesi için gerekli verileri giriniz.")
# Age
Age = st.slider('Age', min_value=20, max_value=100, value=0, step=1)
#Gender
Gender = st.selectbox(
    'How would you like to be contacted?',
    ('F', 'M'))
# Height
Height = st.slider('Height', min_value=120, max_value=220, value=0, step=1)
# Weight
Weight = st.slider('Weight', min_value=20, max_value=180, value=0, step=1)
# Body Fat
Body_Fat = st.slider('Body Fat', min_value=3, max_value=80, value=0, step=1)
# Diastolic
Diastolic = st.slider('Diastolic', min_value=0, max_value=200, value=0, step=1)
# Systolic
Systolic = st.slider('Systolic', min_value=-50, max_value=200, value=0, step=1)
# Grip Force
Grip_Force = st.slider('Grip Force', min_value=0, max_value=200, value=0, step=1)

# Sit and Bend Forward
Sit_and_Bend_Forward = st.slider('Sit and Bend Forward', min_value=0, max_value=200, value=0, step=1)
# Sit-ups Counts
Sit_ups_Counts = st.slider('Sit-ups Counts', min_value=0, max_value=200, value=0, step=1)
# Broad Jump
Broad_Jump = st.slider('Broad Jump', min_value=0, max_value=200, value=0, step=1)

#Encoder ve normalizer seçiniz
st.write("### Verileri nasıl normalize etmek istersiniz?")
encoder = st.selectbox("Encoder seçiniz:", ['OneHotEncoder', 'LabelEncoder'],
                        help='Encoder türünü belirleyip veriyi ona göre eğitiniz.')
normalizer = st.selectbox("Normalizer seçiniz:", ['MinMaxScaler', 'StandardScaler'],
                        help='Normalizer türünü belirleyip veriyi ona göre eğitiniz.')

if encoder == 'OneHotEncoder':
    encoder = OneHotEncoder()
elif encoder == 'LabelEncoder':
    encoder = LabelEncoder()

if normalizer == 'MinMaxScaler':
    normalizer = MinMaxScaler()
elif normalizer == 'StandardScaler':
    normalizer = StandardScaler()


if st.button('Predict'):

    all_data = pd.read_csv("/Users/erem/Documents/YL/Web Madenciliği/stremalit-sample/bodyPerformance.csv")



    data = np.array([[Age,Gender, Height, Weight, Body_Fat, Diastolic, Systolic, Grip_Force, Sit_and_Bend_Forward, Sit_ups_Counts, Broad_Jump,"A"]])
    data_created = pd.DataFrame(data, columns=["age", "gender", "height_cm", "weight_kg", "body fat_%", "diastolic", "systolic", "gripForce", "sit and bend forward_cm", "sit-ups counts", "broad jump_cm","class"])
    #add genereated data to all data
    all_data = all_data.append(data_created, ignore_index=True)
    #preprocess

    prep = Preprocess(all_data, encoder, normalizer)
    data, label = prep.preprocess()

    #get last row
    data = data[-1:]
    data = data.drop(["class"], axis=1)
    #get last row
    label = label[-1:]

    #predict
    prediction = model.predict(data)
    st.write("### Tahmin sonucu: ", prediction[0])


    #
    st.write("### Verileriniz: ", data)
    #predict
    # prediction = model.predict()
    # st.write("### Tahmin sonucu: ", prediction[0])