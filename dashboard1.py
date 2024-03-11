import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import pickle
from sklearn import  metrics
import numpy as np

#Define CSS style
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

#Load data
df = pd.read_csv('forecast_data.csv')
df['Date'] = pd.to_datetime (df['Date']) # create a new column 'data time' of datetime type
#df = df.set_index('Date') # make 'datetime' into index
#df.rename(columns = {'Power-1':'power', 'Day week':'day'}, inplace = True)
df2=df.iloc[:,1:5]
X2=df2.values
fig = px.line(df, x="Date", y=df.columns[1:4])


df_real = pd.read_csv('real_results.csv')
y2=df_real['Power (kW) [Y]'].values

#Load and run models

with open('LR_model.pkl','rb') as file:
    LR_model2=pickle.load(file)

y2_pred_LR = LR_model2.predict(X2)



#Evaluate errors
MAE_LR=metrics.mean_absolute_error(y2,y2_pred_LR) 
MSE_LR=metrics.mean_squared_error(y2,y2_pred_LR)  
RMSE_LR= np.sqrt(metrics.mean_squared_error(y2,y2_pred_LR))
cvRMSE_LR=RMSE_LR/np.mean(y2)



#Load RF model
with open('RF_model.pkl','rb') as file:
    RF_model2=pickle.load(file)

y2_pred_RF = RF_model2.predict(X2)

#Evaluate errors
MAE_RF=metrics.mean_absolute_error(y2,y2_pred_RF) 
MSE_RF=metrics.mean_squared_error(y2,y2_pred_RF)  
RMSE_RF= np.sqrt(metrics.mean_squared_error(y2,y2_pred_RF))
cvRMSE_RF=RMSE_RF/np.mean(y2)

d = {'Methods': ['Linear Regression','Random Forest'], 'MAE': [MAE_LR, MAE_RF], 'MSE': [MSE_LR, MSE_RF], 'RMSE': [RMSE_LR, RMSE_RF],'cvMSE': [cvRMSE_LR, cvRMSE_RF]}
df_metrics = pd.DataFrame(data=d)
d={'Date':df_real['Date'].values, 'LinearRegression': y2_pred_LR,'RandomForest': y2_pred_RF}
df_forecast=pd.DataFrame(data=d)
df_results=pd.merge(df_real,df_forecast, on='Date')


fig2 = px.line(df_results,x=df_results.columns[0],y=df_results.columns[1:4])

# Define auxiliary functions
def generate_table(dataframe, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server=app.server

app.layout = html.Div([
    html.H2('IST Energy Forecast tool (kWh)'),
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Raw Data', value='tab-1'),
        dcc.Tab(label='Forecast', value='tab-2'),
    ]),
    html.Div(id='tabs-content')
])

@app.callback(Output('tabs-content', 'children'),
              Input('tabs', 'value'))

def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            html.H3('IST Raw Data'),
            dcc.Graph(
                id='yearly-data',
                figure=fig,
            ),
            
        ])
    elif tab == 'tab-2':
        return html.Div([
            html.H3('IST Electricity Forecast (kWh)'),
            dcc.Graph(
                id='yearly-data',
                figure=fig2,
                ),
            generate_table(df_metrics)
        ])


if __name__ == '__main__':
    app.run_server(debug=False)
