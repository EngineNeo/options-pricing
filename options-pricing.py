import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import datetime as dt
from datetime import datetime, timedelta
from arch.univariate import arch_model
from scipy.stats import norm
import math
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Streamlit setup
st.set_page_config(
    page_title="Options Theoretical Pricing",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Collecting Data
st.title('SPY Options Pricing Tool')
@st.cache_data
def fetch_data(start_date):
  data = yf.download("SPY", interval="1d", start=start_date)
  return data

# Sidebar
st.sidebar.header('User Input Parameters')
start_date = st.sidebar.date_input("Start Date", value=(datetime.today() - timedelta(days=59)))
data = fetch_data(start_date)

# Underlying Price Chart
st.subheader('Underlying Price Chart')
fig_underlying = px.line(data, x=data.index, y='Adj Close', title='SPY Adjusted Close Price')
st.plotly_chart(fig_underlying)

# Volatility Calculation and Chart
st.subheader('Volatility Forecast')
returns = 10 * data['Adj Close'].pct_change().dropna()
am = arch_model(returns, vol="garch", p=2, o=0, q=2, dist="Normal")
res = am.fit(update_freq=5)
horizon = st.sidebar.slider('Forecast Horizon', min_value=1, max_value=10, value=5, step=1)
forecasts = res.forecast(horizon=horizon)
variance = forecasts.variance.dropna()
volatility = np.sqrt(variance)
plot_data = volatility.T
plot_data_melted = pd.melt(plot_data.reset_index(), id_vars=['index'], var_name='horizon', value_name='volatility')

fig_volatility = px.line(plot_data_melted, x='index', y='volatility', color='horizon', title="Forecasted Volatility")
fig_volatility.update_xaxes(title_text='Time')
fig_volatility.update_yaxes(title_text='Volatility')
st.plotly_chart(fig_volatility)

# Implied Options Prices
st.subheader('Implied Options Prices')
expiration_date = st.sidebar.date_input(
    "Expiration Date", value=(datetime.today() + timedelta(days=2)))
spy_options = yf.Ticker("SPY").option_chain(
    expiration_date.strftime('%Y-%m-%d'))

option_type = st.radio("Option Type", ("Calls", "Puts"))  # Radio

if option_type == "Calls":
    selected_options = spy_options.calls
else:
    selected_options = spy_options.puts

selected_options['contract_name'] = selected_options.apply(
    lambda row: f"{int(row['strike'])}{option_type[0].upper()}", axis=1)

selected_options['bid'] = selected_options['bid'].astype(float)
selected_options['ask'] = selected_options['ask'].astype(float)
selected_options['change'] = selected_options['change'].astype(float)
selected_options['percentChange'] = selected_options['percentChange'].astype(
    float)
selected_options['openInterest'] = selected_options['openInterest'].astype(
    float)

selected_options = selected_options[['contract_name', 'lastPrice', 'bid', 'ask',
                                     'change', 'percentChange', 'volume', 'openInterest', 'impliedVolatility']]

# Display the table with pagination using st.dataframe
st.dataframe(selected_options, height=300)

# Theoretical Options Pricing
st.subheader('Theoretical Options Pricing')
recent_price = yf.download("SPY", period="1d")
underlying = recent_price['Adj Close'].iloc[0]
interest_price = yf.download("^TNX", period="1d")
interest_price = interest_price['Adj Close'].iloc[0] * .01
vol = volatility.iloc[0][0]
time_to_expr = (pd.to_datetime(expiration_date) - pd.to_datetime(datetime.today().date())).days
time = time_to_expr/252

def d1(underlying, strike, interest, vol, time):
  return (np.log(underlying/strike) + (interest+(math.pow(vol, 2)/2))*time)/(vol*np.sqrt(time))

def d2(d1, vol, time):
  return d1 - vol*np.sqrt(time)

def call(underlying, d1, d2, strike, interest, time):
  return underlying * norm.cdf(d1) - strike * np.exp(-interest*time) * norm.cdf(d2)

def put(underlying, call, strike, interest, time):
  return call - underlying + strike*np.exp(-interest*time)

def delta_call(d1):
  return norm.cdf(d1)

def delta_put(d1):
  return norm.cdf(d1) - 1

def gamma(underlying, d1, vol, time):
  return norm.pdf(d1) / (underlying * vol * np.sqrt(time))

def vega(underlying, d1, time):
  return underlying * norm.pdf(d1) * np.sqrt(time) / 100

def theta_call(underlying, d1, d2, strike, interest, vol, time):
    return (-((underlying * norm.pdf(d1) * vol) / (2 * np.sqrt(time))) - (interest * strike * np.exp(-interest * time) * norm.cdf(d2)))/365

def theta_put(underlying, d1, d2, strike, interest, vol, time):
    return (-((underlying * norm.pdf(d1) * vol) / (2 * np.sqrt(time))) + (interest * strike * np.exp(-interest * time) * norm.cdf(-d2)))/365

def rho_call(strike, d2, interest, time):
  return strike * time * np.exp(-interest * time) * norm.cdf(d2) / 100

def rho_put(strike, d2, interest, time):
  return -strike * time * np.exp(-interest * time) * norm.cdf(-d2) / 100

# Retrieve actual options prices using yfinance
spy_options = yf.Ticker("SPY").option_chain(expiration_date.strftime('%Y-%m-%d'))
actual_calls = spy_options.calls
actual_puts = spy_options.puts

# Find the closest strike prices to the underlying price
closest_strike_call = actual_calls.iloc[(actual_calls['strike'] - underlying).abs().argsort()[:1]]['strike'].values[0]

closest_strike_put = actual_puts.iloc[(actual_puts['strike'] - underlying).abs().argsort()[:1]]['strike'].values[0]

selected_strike = st.sidebar.slider("Select Strike Price", min_value=int(actual_calls['strike'].min(
)), max_value=int(actual_calls['strike'].max()), step=1, value=int(closest_strike_call))

# Calculating theoretical pricing and greeks for the selected strike price
d1_call = d1(underlying, selected_strike, interest_price, vol, time)
d2_call = d2(d1_call, vol, time)
theoretical_call = call(underlying, d1_call, d2_call,
                        selected_strike, interest_price, time)
delta_call_value = delta_call(d1_call)
gamma_call_value = gamma(underlying, d1_call, vol, time)
vega_call_value = vega(underlying, d1_call, time)
theta_call_value = theta_call(
    underlying, d1_call, d2_call, selected_strike, interest_price, vol, time)
rho_call_value = rho_call(selected_strike, d2_call, interest_price, time)

d1_put = d1(underlying, selected_strike, interest_price, vol, time)
d2_put = d2(d1_put, vol, time)
theoretical_put = put(underlying, theoretical_call,
                      selected_strike, interest_price, time)
delta_put_value = delta_put(d1_put)
gamma_put_value = gamma(underlying, d1_put, vol, time)
vega_put_value = vega(underlying, d1_put, time)
theta_put_value = theta_put(
    underlying, d1_put, d2_put, selected_strike, interest_price, vol, time)
rho_put_value = rho_put(selected_strike, d2_put, interest_price, time)

# Retrieve actual options prices for the selected strike price
actual_call_price = actual_calls[actual_calls['strike']
                                 == selected_strike]['lastPrice'].values[0]
actual_put_price = actual_puts[actual_puts['strike']
                               == selected_strike]['lastPrice'].values[0]

# Create a DataFrame for the table
table_data = pd.DataFrame({
    'Metric': ['Theoretical Price', 'Actual Price', 'Delta', 'Gamma', 'Vega', 'Theta', 'Rho'],
    'Call Option': [
        f"{theoretical_call:.2f}",
        f"{actual_call_price:.2f}",
        f"{delta_call_value:.2f}",
        f"{gamma_call_value:.2f}",
        f"{vega_call_value:.2f}",
        f"{theta_call_value:.2f}",
        f"{rho_call_value:.2f}"
    ],
    'Put Option': [
        f"{theoretical_put:.2f}",
        f"{actual_put_price:.2f}",
        f"{delta_put_value:.2f}",
        f"{gamma_put_value:.2f}",
        f"{vega_put_value:.2f}",
        f"{theta_put_value:.2f}",
        f"{rho_put_value:.2f}"
    ]
})

# Set the 'Metric' column as the index
table_data.set_index('Metric', inplace=True)

# Display the table using Streamlit
st.table(table_data)

# Create a DataFrame for call options data
call_data = pd.DataFrame({
    'Strike': [selected_strike],
    'Theoretical Price': [theoretical_call],
    'Actual Price': [actual_call_price],
    'Delta': [delta_call_value],
    'Gamma': [gamma_call_value],
    'Vega': [vega_call_value],
    'Theta': [theta_call_value],
    'Rho': [rho_call_value]
})

# Create a DataFrame for put options data
put_data = pd.DataFrame({
'Strike': [selected_strike],
'Theoretical Price': [theoretical_put],
'Actual Price': [actual_put_price],
'Delta': [delta_put_value],
'Gamma': [gamma_put_value],
'Vega': [vega_put_value],
'Theta': [theta_put_value],
'Rho': [rho_put_value]

})

# Melt the DataFrames for easier plotting
call_data_melted = pd.melt(call_data, id_vars=['Strike'], var_name='Metric', value_name='Value')
put_data_melted = pd.melt(put_data, id_vars=['Strike'], var_name='Metric', value_name='Value')

# Create a DataFrame for the actual vs theoretical prices
price_data = pd.DataFrame({
    'Option Type': ['Call', 'Put'],
    'Actual Price': [actual_call_price, actual_put_price],
    'Theoretical Price': [theoretical_call, theoretical_put]
})

# Bar Chart
fig_prices = px.bar(price_data, x='Option Type', y=['Actual Price', 'Theoretical Price'],
                    barmode='group', title=f'Actual vs Theoretical Prices (Strike: {selected_strike})')

fig_prices.update_layout(
    xaxis_title='Option Type',
    yaxis_title='Price',
    legend_title='Price Type',
    height=400,
    width=800
)

# Display the chart
st.plotly_chart(fig_prices)
