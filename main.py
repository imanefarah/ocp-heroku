import pandas as pd
import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import xgboost
from sklearn.linear_model import Lasso,Ridge
from sklearn.linear_model import LassoCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import plotly.express as px


def app():
  padding = 0
  st.markdown(f""" <style>
    .reportview-container .main .block-container{{
        padding-top: {padding}rem;
        padding-right: {padding}rem;
        padding-left: {padding}rem;
        padding-bottom: {padding}rem;
    }} </style> """, unsafe_allow_html=True)



# Download CSV data
  def filedownload(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download={filename}>Download {filename} File</a>'
    return href


  with st.sidebar.header(' Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

# Displays the dataset
  st.subheader('** Dataset**')
  if uploaded_file is not None:
    df = pd.read_csv(uploaded_file,index_col=0)

  st.sidebar.header(' Select the method')
  button_method=st.sidebar.radio('',['Embedded method','Choose lags'])


  def get_lags():
    df3 = df.copy()
    for col in df3.columns:
        for lag in lags:
            df3[col + "_" + str(lag)] = df3[col].diff(periods=lag)
    return df3
  if button_method == 'Choose lags':
    lags = st_tags_sidebar(
    label='# Enter Lags:',
    text='Press enter to add more')
    df3=get_lags()
    X_last = df3.tail(1)
    df3["y"] = df3["DAP _1"].shift(-1)
    df3 = df3.dropna()


  if button_method == 'Embedded method':
    df2=df.copy()
    for col in df2.columns:
        for lag in range(1, 13):
            df2[col + "_" + str(lag)] = df2[col].diff(periods=lag)
    X_last = df2.tail(1)
    df2["y"] = df2["DAP _1"].shift(-1)
    df2 = df2.dropna()
    df4=df2[:-10]
    X = df4.drop('y', axis=1)
    Y = df4['y']
    reg = LassoCV()
    reg.fit(X, Y)
    coef = pd.Series(reg.coef_, index=X.columns)
    d = coef[coef != 0]
    cols = d.index
    df3 = df2[cols]
    X_last = X_last[cols].tail(1)
    df3['y'] = df2['y'].values


  features=df3.columns
# Sidebar - feature selection
  st.sidebar.header(' select feature')
  container = st.sidebar.beta_container()
  all = st.sidebar.checkbox("Select all")
  if all:
    selected_feature = container.multiselect("",list(features),list(features))
  else:
    selected_feature = container.multiselect("", features)

  df3=df3[selected_feature]

  st.sidebar.header(' select regressor')
  Regressor_name = st.sidebar.selectbox(
    '',
    ('XGBOOST', 'RandomForestRegressor', 'Lasso')
   )

  def get_Regressor(model_name):
    model = None
    if model_name == 'XGBOOST':
        model =  xgboost.XGBRegressor(n_estimators=50, max_depth=5, eta=0.1, subsample=0.7, colsample_bytree=0.8)
    elif model_name == 'RandomForestRegressor':
        model = RandomForestRegressor(n_estimators=70)
    else:
        model = model = Lasso(alpha=0.9)
    return model
  model=get_Regressor(Regressor_name )

#### Regression####
# Sidebar - Specify parameter settings

  with st.sidebar.header('test size'):
    split_size = st.sidebar.slider('% for Test Set',0.1,0.9,0.2,0.05)
  dff = df3[:-10]
  X = np.array(dff.iloc[:,:-1] )# Using all column except for the last column as X#
  Y = np.array(dff.iloc[:,-1]).reshape(-1,1)# Selecting the last column as Y
  X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=split_size, shuffle=False)
  f1,f2,f3,f4=st.beta_columns(4)
  if f1.button('show dataset'):
   st.write(df3)
  if f2.button('Dataset dimension'):
     a1, a2, a3,a4,a5= st.beta_columns(5)
     a1.write('dataset')
     a1.info(df3.shape)
     with a2:
         a2.write('X_train')
         a2.info(X_train.shape)
     a3.write('y_train')
     a3.info(y_train.shape)
     a4.write('X_test')
     a4.info(X_test.shape)
     a5.write('y_test')
     a5.info(y_test.shape)


#visualization
  if f3.button('Data visualization'):
    plot_name = st.selectbox('Select variable', (df3.columns) )
    fig=px.line(df3[f'{plot_name}'], x=df3.index, y=f'{plot_name}')
    fig.update_traces(line_color='#272727')
    fig.update_xaxes(rangeslider_visible=True)
    st.plotly_chart(fig)
  if f4.button('Explore data'):
    advert_report = sv.analyze(df)
    advert_report.show_html('Advertising.html')

#Model
  st.markdown('### ** Model**')
  st.write(f'Regressor = {Regressor_name}')



  def build_testmodel(df):
    st.markdown('### **Score on test dataset**')
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    score = round(r2_score(y_test, y_pred),3)
    MSE = round(mean_squared_error(y_test, y_pred),3)
    b1, b2 = st.beta_columns(2)
    b1.write('r2_score')
    b1.info(score)
    with b2:
        b2.write('MSE')
        b2.info(MSE)
    st.markdown('### **Score on validation dataset**')
  val=df3[-10:]
  X_val=np.array(val.iloc[:,:-1])
  Y_val=np.array(val.iloc[:,-1]).reshape(-1,1)
  def build_valmodel(X_val,Y_val):
    model.fit(X,Y)
    y_pred=model.predict(X_val)
    score = round(r2_score(Y_val, y_pred),3)
    MSE = round(mean_squared_error(Y_val, y_pred),3)
    b1, b2 = st.beta_columns(2)
    b1.write('r2_score')
    b1.info(score)
    with b2:
        b2.write('MSE')
        b2.info(MSE)
    st.markdown('### ** Model Prediction**')
    c1, c2 = st.beta_columns(2)
    prediction = model.predict(X_last.values) + X_last['DAP '].values[0]
    c1.markdown('### Last DAP')
    c1.info(X_last['DAP '].values[0])
    with c2:
         c2.write('Predicted DAP')
         c2.info(prediction)
    st.markdown('### **Validation Data comparaison**')
    fig, ax = plt.subplots(figsize=(16,4))
    ax.plot(val.iloc[:,-1].index, y_pred.cumsum(),label="DAP prediction cumulative variation")
    ax.plot(val.iloc[:,-1].index, val.iloc[:,-1].cumsum(),label="DAP cumulative variation")
    ax.legend()
    st.pyplot(plt)

  def build_walktestmodel(df):
    st.markdown('### **Score on test dataset**')
    #Walk forward
    df_X = X_train
    df_Y = y_train
    y = y_test
    x = X_test
    predictions = []
    for i in range(len(y)):
        model.fit(df_X, df_Y )
        pred = model.predict(X_test)[i]
        predictions = np.append(predictions, pred)
        obs = y[i]
        df_Y = np.append(df_Y, obs)
        df_X = np.vstack([df_X, x[i]])
    score = r2_score(y_test, predictions)
    MSE=mean_squared_error(y_test, predictions)


    b1, b2, b3,b4= st.beta_columns(4)
    b1.write('r2_score')
    b1.info(score)
    with b2:
       b2.write('MSE')
       b2.info(MSE)
    st.markdown('### **Score on validation dataset**')


  def build_walkvalmodel(val):
    df_X = X
    df_Y = Y
    y = Y_val
    x = X_val
    predictions = []
    for i in range(len(y)):
        model.fit(df_X, df_Y)
        pred = model.predict(X_val)[i]
        predictions = np.append(predictions, pred)
        obs = y[i]
        df_Y = np.append(df_Y, obs)
        df_X = np.vstack([df_X, x[i]])
    score = r2_score(Y_val, predictions)
    MSE = mean_squared_error(Y_val, predictions)
    b1, b2= st.beta_columns(2)

    b1.write('r2_score')
    b1.info(score)
    with b2:
       b2.write('MSE')
       b2.info(MSE)
    st.markdown('### ** Model Prediction**')
    c1, c2 = st.beta_columns(2)
    prediction = model.predict(X_last.values) + X_last['DAP '].values[0]
    c1.write(f'last DAP')
    c1.info(X_last['DAP '].values[0])
    with c2:
        c2.write(f'Predicted DAP')
        c2.info(prediction)
    st.markdown('### **Validation Data comparaison**')
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.plot(val.iloc[:, -1].index, predictions.cumsum(),label=f"DAP prediction cumulative variation")
    ax.plot(val.iloc[:, -1].index, val.iloc[:, -1].cumsum(), label=f"DAP cumulative variation")
    ax.legend()
    st.pyplot(plt)

  d1, d2 = st.beta_columns(2)
  if d1.button('work with walk forward method'):
    build_walktestmodel(dff)
    build_walkvalmodel(val)
  if d2.button('work with classic method'):
    build_testmodel(dff)
    build_valmodel(X_val,Y_val)


  padding = 0
  st.markdown(f""" <style>
    .reportview-container .main .block-container{{
        padding-top: {padding}rem;
        padding-right: {padding}rem;
        padding-left: {padding}rem;
        padding-bottom: {padding}rem;
    }} </style> """, unsafe_allow_html=True)


