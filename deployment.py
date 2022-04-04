#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

st.title('Model Deployment: Random Forest Regressor')

st.header('User Input Parameters')

manufac_dict={'Acura':0,'AlfaRomeo':1,'Audi':2,'BMW':3,'Buick':4,'Cadillac':5,'Chery':6,'Chevrolet':7,'Chrysler':8,'Citroen':9,'Dacia':10,'Daewoo':11,'Dodge':12,'Fiat':13,'Ford':14,'Geely':15,'GreatWall':16,'Honda':17,'Hyundai':18,'Infiniti':19,'Iveco':20,'Jaguar':21,'Jeep':22,'Kia':23,'LADA':24,'Lancia':25,'LandRover':26,'Lexus':27,'Lifan':28,'Lincoln':29,'Mazda':30,'Mercedes-Benz':31,'Mini':32,'Mitsubishi':33,'Nissan':34,'Opel':35,'Peugeot':36,'Pontiac':37,'Porsche':38,'Renault':39,'Rover':40,'Saab':41,'Seat':42,'Skoda':43,'SsangYong':44,'Subaru':45,'Suzuki':46,'Toyota':47,'Volkswagen':48,'Volvo':49,'ВАЗ':50,'ГАЗ':51,'ЗАЗ':52,'Москвич':53,'УАЗ':54}

color_dict={'black':0,'blue':1,'brown':2,'green':3,'grey':4,'orange':5,'other':6,'red':7,'silver':8,'violet':9,'white':10,'yellow':11}

body_dict={'cabriolet':0,'coupe':1,'hatchback':2,'liftback':3,'limousine':4,'minibus':5,'minivan':6,'pickup':7,'sedan':8,'suv':9,'universal':10,'van':11}

d_dict={'all':0,'front':1,'rear':2}
t_dict={"auto":0,"manual":1}

manufac_list=["Acura","AlfaRomeo","Audi","BMW","Buick","Cadillac","Chery","Chevrolet","Chrysler","Citroen","Dacia","Daewoo","Dodge","Fiat","Ford","Geely","GreatWall","Honda","Hyundai","Infiniti","Iveco","Jaguar","Jeep","Kia","LADA","Lancia","LandRover","Lexus","Lifan","Lincoln","Mazda","Mercedes-Benz","Mini","Mitsubishi","Nissan","Opel","Peugeot","Pontiac","Porsche","Renault","Rover","Saab","Seat","Skoda","SsangYong","Subaru","Suzuki","Toyota","Volkswagen","Volvo","ВАЗ","ГАЗ","ЗАЗ","Москвич","УАЗ"]
color_list=["black","blue","brown","green","grey","orange","other","red","silver","violet","white","yellow"]
body_list=["cabriolet","coupe","hatchback","liftback","limousine","minibus","minivan","pickup","sedan","suv","universal","van"]
d_list=["all","front","rear"]
t_list=["auto","manual"]




manufacturer=st.selectbox(label='Select your Car Manufacturer', options=manufac_list)
manufacturer_name=manufac_dict[manufacturer]

transmission=st.selectbox(label='Select Transmission', options=t_list)
transmissions=t_dict[transmission]

color=st.selectbox(label='Select your favourite Car Colour', options=color_list)
colors=color_dict[color]

Odometer_value=st.number_input("Insert KM Driven")

Year_produced=st.slider("Enter the year", 1970, 2021)

engine_capacity=st.slider("Enter Engine Size",0,7)

body=st.selectbox(label='Type of car', options=body_list)
body_types=body_dict[body]

drive=st.selectbox(label='Drivetrain', options=d_list)
drivetrains=d_dict[drive]

No_of_photos=st.slider("Insert No. of Photos Uploaded",0,20)

cars=pd.read_csv("cars_data.csv")
final=pd.DataFrame(cars[['manufacturer_name', 'transmission', 'color', 'odometer_value', 'year_produced','engine_capacity', 'body_type','drivetrain','number_of_photos','price_usd']])
final=final.dropna()
c=[ 'manufacturer_name','transmission', 'color','body_type','drivetrain']
final[c]=final[c].apply(LabelEncoder().fit_transform)

X=final.iloc[:,0:9]
Y=final.iloc[:,9]
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.3, random_state=0)

rfr=RandomForestRegressor()
rfr.fit(X_train.values, y_train.values)

random_forest_score =rfr.score(X_test.values, y_test.values)
column_data = X.columns.values

def predict_price_forest(manufacturer_name,transmissions,colors,Odometer_value,Year_produced,engine_capacity,body_types,drivetrains,No_of_photos):
        prediction=rfr.predict([[manufacturer_name,transmissions,colors,Odometer_value,Year_produced,engine_capacity,body_types,drivetrains,No_of_photos]])

        return prediction
        
result=0
st.button('Price Prediction')
result=predict_price_forest(manufacturer_name,transmissions,colors,Odometer_value,Year_produced,engine_capacity,body_types,drivetrains,No_of_photos)
st.success('The Price of Car in( Belarusian Ruble(BYN)) is {}'.format(result))

