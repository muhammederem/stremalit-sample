import streamlit as st
import pandas as pd
import numpy as np
from preprocess import Preprocess
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from matplotlib import pyplot as plt
from xgboost import XGBClassifier
import pickle


def plot_confusion_matrix(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(cm)
    ax.grid(False)
    ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
    ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
    ax.set_ylim(1.5, -0.5)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
    plt.show()


data = pd.read_csv("/Users/erem/Documents/YL/Web Madenciliği/stremalit-sample/bodyPerformance.csv")
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


is_importance = False
is_correlation = False
is_outlier = False

if st.checkbox("Sütun önemine göre veri temizleme yapmak istiyorum."):
    is_importance = True

if st.checkbox("Korelasyona göre veri temizleme yapmak istiyorum."):
    is_correlation = True

if st.checkbox("Veri setindeki outlierlerı elemek istiyorum"):
    is_outlier = True


if  is_importance:
    importance_threshold = st.slider('Importance Threshold', min_value=0.0, max_value=.1, value=0.05, step=0.005)
if is_correlation:
    corr_threshold = st.slider('Correlation Threshold', min_value=0.0, max_value=1.0, value=0.5, step=0.1)



if is_importance or is_correlation or is_outlier:



    st.write("Veri ön işleme başarıyla tamamlandı.")
    st.write("## Eğitim için Model Seçimi")
    st.write("### Model seçiniz.")
    model_name = st.selectbox("Model seçiniz:", ['Random_Forest', 'SVC', 'XGBoost'],
                        help='Model türünü belirleyip veriyi ona göre eğitiniz.')





    if model_name == 'Random_Forest':
        st.write("Random Forest modeli seçildi.")
        st.write("### Model parametrelerini belirleyiniz.")
        #en önemli 5 parametre
        n_estimators = st.slider('n_estimators', min_value=100, max_value=1000, value=100, step=100)
        max_depth = st.slider('max_depth', min_value=2, max_value=10, value=2, step=1)
        max_features = st.slider('max_features', min_value=2, max_value=10, value=2, step=1)
        min_samples_leaf = st.slider('min_samples_leaf', min_value=2, max_value=10, value=2, step=1)
        train_size = st.slider('train_size', min_value=0.0, max_value=1.0, value=0.7, step=0.1)
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features, min_samples_leaf=min_samples_leaf)

    elif model_name == 'XGBoost':
        st.write("XGBoost modeli seçildi.")
        st.write("### Model parametrelerini belirleyiniz.")
        #en önemli 5 parametre
        n_estimators = st.slider('n_estimators', min_value=100, max_value=1000, value=100, step=100)
        max_depth = st.slider('max_depth', min_value=2, max_value=10, value=2, step=1)
        max_features = st.slider('max_features', min_value=2, max_value=10, value=2, step=1)
        min_samples_leaf = st.slider('min_samples_leaf', min_value=2, max_value=10, value=2, step=1)
        train_size = st.slider('train_size', min_value=0.0, max_value=1.0, value=0.7, step=0.1)
        model = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features, min_samples_leaf=min_samples_leaf)


    elif model_name == 'SVC':
        st.write("SVC modeli seçildi.")
        st.write("### Model parametrelerini belirleyiniz.")
        #en önemli 5 parametre
        C = st.slider('C', min_value=0.0, max_value=1.0, value=0.5, step=0.1)
        kernel = st.selectbox("kernel", ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
                        help='kernel türünü belirleyip veriyi ona göre eğitiniz.')
        degree = st.slider('degree', min_value=2, max_value=10, value=2, step=1)
        gamma = st.slider('gamma', min_value=0.0, max_value=1.0, value=0.5, step=0.1)

        train_size = st.slider('train_size', min_value=0.0, max_value=1.0, value=0.7, step=0.1)
        model = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma)

        

    if st.button("Eğitimi başlat"):
        data, target = Preprocess(data,encoder,scaler).preprocess()
        #drop class from data
        data = data.drop('class', axis=1)


        st.write("### Veri setinin ilk 5 satırı:")
        st.write(data.head())

        st.write("### Target ilk 5 satırı:")
        st.write(target.head())
        
        
        x_train, x_test, y_train, y_test = train_test_split(data, target, train_size=train_size, random_state=42)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        #save model
        pickle.dump(model, open(model_name+'_model.pkl', 'wb'))

        st.write("### Modelin performansı:")
        st.write("Accuracy: ", accuracy_score(y_test, y_pred))
        st.write("Confusion Matrix: ", confusion_matrix(y_test, y_pred))
        st.write("Classification Report: ", classification_report(y_test, y_pred))

        #tekil değerlerin tahmin edilmesi