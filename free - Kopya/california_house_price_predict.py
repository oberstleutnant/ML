# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 19:49:40 2023

@author: emreu
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn

from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("price_house.csv")
	

#null veriyi bulma

null = data.isnull().sum()
#data temizleme ve düzenleme
data.pop("İlan_Numarası")
data.pop("İlan_Güncelleme_Tarihi")
data.pop("İlan_Oluşturma_Tarihi")
data.pop("Takas")
data.pop("Fiyat_Durumu")
data.pop("Kategorisi")
"""
#veriyi görselleştirme
#tam olmadı tekrar bak!!!!
plt.figure(figsize=(1,100))

sbn.displot(data["Fiyatı"])
sbn.countplot(data["Binanın_Yaşı"])
sbn.scatterplot(x="Binanın_Yaşı",y="Fiyatı",data=data)
"""
#korealasyon verilerin ne şekilde etkilendiği corr
corr = data.corr()["Fiyatı"].sort_values()
#data temizleme ve düzenleme
data2 = data.iloc[:,[0,2,8,14,19,20]]
data3 = data.iloc[:,[1,3,4,5,6,7,9,10,11,12,13,15,16,17,18]]
datab = pd.concat([data2,data3],axis=1)

#verinin kategorikaldan sayısal veriye dönüşümü
data4= pd.DataFrame()
print(data4)
for key in datab.keys():
    if datab[key].dtype == "object":
        new_key = key + "_cat"
        data4[new_key] = datab[key].astype("category").cat.codes
        
    else:
        data4[key] = datab[key]
        
from sklearn.model_selection import train_test_split
X=data4.drop(["Fiyatı"],axis=1)
Y=data4["Fiyatı"]

x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.3,random_state=15)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.callbacks import EarlyStopping

model = Sequential()

model.add(Dense(units=30,activation = "relu"))
model.add(Dense(units=15,activation = "relu"))
model.add(Dense(units=15,activation = "relu"))
model.add(Dense(units=1,activation = "sigmoid"))

model.compile(loss="binary_crossentropy",optimizer = "adam")

model.fit(x=x_train, y=y_train, epochs=700,validation_data=(x_test,y_test),verbose=1)

model.history.history
modelKaybi = pd.DataFrame(model.history.history)
print(modelKaybi.plot())