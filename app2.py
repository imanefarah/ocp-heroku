import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import xgboost
from sklearn.linear_model import Lasso,Ridge
from sklearn.linear_model import LassoCV
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import plotly.express as px
html_temp= """
    <div style="background-color:#464e5f;padding:10px;border-radius:10">
    <h1 style="color:white;text-align:center;">
    </div>
    """

def app():
  [theme]
  primaryColor = "#579D02"
  backgroundColor = "#FFFFFF"
  secondaryBackgroundColor = "##272727"
  textColor = "#272727"
  font = "sans serif"
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



  df2=df.copy()
  for col in df2.columns:
        for lag in range(1, 13):
            df2[col + "_" + str(lag)] = df2[col].diff(periods=lag)

  df2["y"] = df2["DAP _1"].shift(-1)
  df2 = df2.dropna()
  X = df2.drop('y', axis=1)
  Y = df2['y']
  reg = LassoCV()
  reg.fit(X, Y)
  coef = pd.Series(reg.coef_, index=X.columns)
  d = coef[coef != 0]
  cols = d.index
  df3 = df2[cols]
  df3['y'] = df2['y'].values


  features=df3.columns
# Sidebar - feature selection
  container = st.sidebar.beta_container()
  all = st.sidebar.checkbox("Select all")
  if all:
    selected_feature = container.multiselect(" Select features:",list(features),list(features))
  else:
    selected_feature = container.multiselect(" Select features:", features)

  df3=df3[selected_feature]




  X_last = df3.tail(1).drop('y',1)

  X = np.array(df3.iloc[:,:-1] )# Using all column except for the last column as X#
  Y = np.array(df3.iloc[:,-1]).reshape(-1,1)# Selecting the last column as Y
  X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)
  f1,f2,f3,f4=st.beta_columns(4)
  if f1.button('show dataset'):
      components.html(html_temp)
      st.write(df3())
  if f2.button('Dataset dimension'):
     components.html(html_temp)
     a1, a2, a3,a4,a5= st.beta_columns(5)
     a1.write('dataset')
     a1.info(df.shape)
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
    components.html(html_temp)
    plot_name = st.selectbox('Select variable', (df3.columns) )
    fig=px.line(df3[f'{plot_name}'], x=df3.index, y={plot_name})
    fig.update_xaxes(rangeslider_visible=True)
    st.plotly_chart(fig)
  st.markdown('Last DAP:')
  st.info(X_last['DAP '].values)

#Model
  st.markdown('### Bayesian Ridge')
  df_X = X_train
  df_Y = y_train
  y = y_test
  x = X_test
  forest_predictions = []
  for i in range(len(y)):
        forest_model = BayesianRidge()
        forest_model.fit(df_X, df_Y )
        pred = forest_model.predict(X_test)[i]
        forest_predictions= np.append(forest_predictions, pred)
        obs = y[i]
        df_Y = np.append(df_Y, obs)
        df_X = np.vstack([df_X, x[i]])
  forest_score = r2_score(y_test, forest_predictions)
  forest_mse=mean_squared_error(y_test, forest_predictions)
  b1, b2, b3= st.beta_columns(3)
  b1.write('r2_score')
  b1.info(forest_score)
  with b2:
       b2.write('MSE')
       b2.info(forest_mse)
#model prediction
  forest_prediction = forest_model.predict(X_last.values) + X_last['DAP '].values[0]
  b3.write('Predicted DAP')
  b3.info(forest_prediction)

  st.markdown('### Lasso')
  df_X = X_train
  df_Y = y_train
  y = y_test
  x = X_test
  Lasso_predictions = []
  for i in range(len(y)):
        lasso_model = Lasso(alpha=0.9)
        lasso_model.fit(df_X, df_Y )
        pred = lasso_model.predict(X_test)[i]
        Lasso_predictions = np.append(Lasso_predictions, pred)
        obs = y[i]
        df_Y = np.append(df_Y, obs)
        df_X = np.vstack([df_X, x[i]])
  lasso_score = r2_score(y_test, Lasso_predictions)
  lasso_mse=mean_squared_error(y_test, Lasso_predictions)
  b1, b2, b3= st.beta_columns(3)
  b1.write('r2_score')
  b1.info(lasso_score)
  with b2:
       b2.write('MSE')
       b2.info(lasso_mse)
#model prediction
  lasso_prediction = lasso_model.predict(X_last.values) + X_last['DAP '].values[0]
  b3.write('Predicted DAP')
  b3.info(lasso_prediction)
  st.markdown('### XGBOOST')
  df_X = X_train
  df_Y = y_train
  y = y_test
  x = X_test
  xgb_predictions = []
  for i in range(len(y)):
        xgb_model = xgboost.XGBRegressor(n_estimators=50, max_depth=5, eta=0.1, subsample=0.7, colsample_bytree=0.8)
        xgb_model.fit(df_X, df_Y )
        pred = lasso_model.predict(X_test)[i]
        xgb_predictions = np.append(xgb_predictions, pred)
        obs = y[i]
        df_Y = np.append(df_Y, obs)
        df_X = np.vstack([df_X, x[i]])
  xgb_score = r2_score(y_test, xgb_predictions)
  xgb_mse=mean_squared_error(y_test, xgb_predictions)
  b1, b2, b3= st.beta_columns(3)
  b1.write('r2_score')
  b1.info(xgb_score)
  with b2:
       b2.write('MSE')
       b2.info(xgb_mse)
#model prediction
  xgb_prediction = xgb_model.predict(X_last.values) + X_last['DAP '].values[0]
  b3.write('Predicted DAP')
  b3.info(xgb_prediction)



  padding = 0
  st.markdown(f""" <style>
    .reportview-container .main .block-container{{
        padding-top: {padding}rem;
        padding-right: {padding}rem;
        padding-left: {padding}rem;
        padding-bottom: {padding}rem;
    }} </style> """, unsafe_allow_html=True)





