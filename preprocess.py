import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, StandardScaler


class Preprocess():

    def __init__(self, data,encoder,scaler,importance_threshold,corr_threshold):
        self.encoder = encoder
        self.scaler = scaler
        self.data = data
        self.importence_threshold = importance_threshold
        self.corr_threshold = corr_threshold
    
    
    def max_values(self,data):
        max_values = []
        for column in data.columns:
            max_values.append(data[column].max())

        return max_values

    def min_values(self,data):
        min_values = []
        for column in data.columns:
            min_values.append(data[column].min())

        return min_values

    def get_min_max(self):
        target = self.data['class']
        data = self.data.drop('class', axis=1)

        max_values = self.max_values(data)
        min_values = self.min_values(data)

        return max_values, min_values

    def preprocess(self):
        try:
            data = self.data.copy()
            try:
                if str(type(self.encoder)) == "<class 'sklearn.preprocessing._encoders.OneHotEncoder'>":
                    gender = data[["gender"]]
                    data.pop("gender")
                    gender_data = pd.DataFrame(OneHotEncoder().fit_transform(gender).toarray(),columns=["Female","Male"])
                    data = data.join(gender_data)
                else:
                    print(self.encoder)
                    print(type(self.encoder))
                    data['gender'] = self.encoder.fit_transform(data['gender'])
            except Exception as e:
                print(e)

            target = data['class']
            data = data.drop('class', axis=1)
            #target
            target = LabelEncoder().fit_transform(target)
            #set column as y column
            target = pd.DataFrame(target, columns=['class'])


            data['age'] = self.scaler.fit_transform(data['age'].values.reshape(-1,1))
            data['height_cm'] = self.scaler.fit_transform(data['height_cm'].values.reshape(-1,1))
            data['weight_kg'] = self.scaler.fit_transform(data['weight_kg'].values.reshape(-1,1))
            data['body fat_%'] = self.scaler.fit_transform(data['body fat_%'].values.reshape(-1,1))
            data['diastolic'] = self.scaler.fit_transform(data['diastolic'].values.reshape(-1,1))
            data['gripForce'] = self.scaler.fit_transform(data['gripForce'].values.reshape(-1,1))
            data['sit and bend forward_cm'] = self.scaler.fit_transform(data['sit and bend forward_cm'].values.reshape(-1,1))
            data['sit-ups counts'] = self.scaler.fit_transform(data['sit-ups counts'].values.reshape(-1,1))
            data['broad jump_cm'] = self.scaler.fit_transform(data['broad jump_cm'].values.reshape(-1,1))

            data = pd.concat([data, target], axis=1)

            return data
        except Exception as e:
            print(e)

    def remove_correlated(self,data):
        correlation = data.corr()
        for i in range(len(correlation.columns)):
            for j in range(i):
                if abs(correlation.iloc[i, j]) > self.corr_threshold:
                    colname = correlation.columns[i]
                    data.drop(colname, axis=1, inplace=True)

        return data

    def find_outliers(self,data):
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3-q1
        lower_bound = q1 -(1.5 * iqr)
        upper_bound = q3 +(1.5 * iqr)

        data = data[(data > lower_bound) & (data < upper_bound)]
        data = data.dropna()

        return data

    def most_important_features(self,data, target):
        from sklearn.ensemble import ExtraTreesClassifier
        model = ExtraTreesClassifier()
        model.fit(data, target)
        for feature in zip(data.columns, model.feature_importances_):
            if feature[1] > self.threshold:
                data.drop(feature[0], axis=1, inplace=True)
                
        return data