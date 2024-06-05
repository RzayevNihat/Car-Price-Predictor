import streamlit as st
import pandas as pd
import pickle
import numpy as np
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler

df=pd.read_csv('cc_price.csv')


label_encoder = LabelEncoder()

    
y_novu_encoding = label_encoder.fit_transform(df['Yanacaq növü'])
y_novu_mapping = {name: value for name, value in zip(df['Yanacaq növü'].str.capitalize().tolist(), y_novu_encoding)}

marka_encoding = label_encoder.fit_transform(df['Marka'].str.capitalize())
marka_mapping = {name: value for name, value in zip(df.Marka.str.capitalize().tolist(), marka_encoding)}

model_encoding = label_encoder.fit_transform(df['Model'])
model_mapping = {name: value for name, value in zip(df.Model.str.capitalize().tolist(), model_encoding)}

b_novu_encoding = label_encoder.fit_transform(df['Ban növü'])
b_novu_mapping = {name: value for name, value in zip(df['Ban növü'].str.capitalize().tolist(), b_novu_encoding)}

reng_encoding = label_encoder.fit_transform(df['Rəng'])
reng_mapping = {name: value for name, value in zip(df['Rəng'].str.capitalize().tolist(), reng_encoding)}
	

interface = st.container()


with interface:

    
    st.write("<h1 style='text-align: center;'>Car price predictor</h1>",unsafe_allow_html=True)
    st.write('***')
    st.write('### Enter Vehicle Features')

    marka,model,y_novu = st.columns(spec = [1,1,1])
    

    with marka:
        marka = st.selectbox(label = 'Marka', options =df['Marka'].str.capitalize().sort_values().unique().tolist())
        b_ili=st.number_input(label='Buraxılış ili',min_value=df['Buraxılış ili'].min(),max_value=df['Buraxılış ili'].max())
        
        yurus=st.number_input(label='Yürüş',min_value=df['Yürüş'].min(),max_value=df['Yürüş'].max())
        
        reng=st.selectbox(label='Rəng',options=df['Rəng'].str.capitalize().sort_values().unique().tolist())
        
    with model:
        model = st.selectbox(label = 'Model', options =df[df['Marka'].str.capitalize() == marka]['Model'].str.capitalize().sort_values().unique().tolist())  
        
        at_gucu=st.number_input(label='At gücü',min_value=df['At gücü'].min(),max_value=df['At gücü'].max())
    with y_novu:
        y_novu=st.selectbox(label='Yanacaq növü',options=df['Yanacaq növü'].sort_values().unique().tolist())
        
        muh_hecm=st.selectbox(label='Mühərrikin həcmi',options=df['Mühərrikin həcmi'].sort_values().unique())
        
        b_novu=st.selectbox(label='Ban növü',options=df['Ban növü'].str.capitalize().sort_values().unique().tolist())

        
    st.markdown(body = '***')
   
    df['Yanacaq növü']=y_novu_encoding
    df['Marka'] = marka_encoding
    df['Model'] = model_encoding
    df['Ban növü']=b_novu_encoding    
    df['Rəng']=reng_encoding
    
    
y_novu=y_novu_mapping[y_novu]
marka = marka_mapping[marka]
model = model_mapping[model]
b_novu=b_novu_mapping[b_novu]
reng=reng_mapping[reng]
    
input_features = pd.DataFrame({
    'Mühərrikin həcmi': [muh_hecm],
    'At gücü': [at_gucu],
    'Yanacaq növü': [y_novu],
    'Marka': [marka],
    'Model': [model],
    'Buraxılış ili': [b_ili],
    'Ban növü':[b_novu],
    'Rəng':[reng],         
    'Yürüş':[yurus]
    })

#scaled_features = scale_data(input_features)  # If using a function
# OR
scaled_features = RobustScaler().fit_transform(input_features)  # Direct scaling

#cars_price = model.predict(scaled_features)

st.subheader(body = 'Model Prediction')
    

with open(file='reg_model.pickle', mode='rb') as pickled_model:
    model = pickle.load(file=pickled_model) 
    
if st.button('Predict'):
    cars_price = model.predict(scaled_features)

    with st.spinner('Sending input features to model...'):
        time.sleep(2)

    st.success('Prediction is ready')
    time.sleep(1)
    st.markdown(f'### Car\'s estimated price is  {int(cars_price)} AZN')
        


