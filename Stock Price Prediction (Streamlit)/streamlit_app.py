# streamlit run streamlit_app.py

#%% Libraries needed
import streamlit as st
import yfinance as yf
import pandas as pd

#%% Title Page
st.title('Predicting Stock Prices using Machine Learning')
st.markdown('---')


# %%
markets_dict = {
    'S&P 500': '^GSPC',
    'Google': '^GOOG'}


left_sb, mid_sb, rght_sb = st.columns([3,3,3])

with left_sb: 

    markets_dict = {
        'S&P 500': '^GSPC', 
        'Google': '^GOOG'}

    index = st.selectbox('Select an index', list(markets_dict.keys()))

   

with mid_sb:
    start_year = st.number_input('Select Start Year (1950-2022)', 1950, 2022)


with rght_sb:

    interval_dict = {
        '1 Day': '1d', 
        '5 Days': '5d', 
        '1 Week': '1wk', 
        '1 Month': '1mo', 
        '3 Months': '3mo'}

    interval = st.selectbox('Select Interval', list(interval_dict.keys()))

sp500 = yf.Ticker("^GSPC")
sp500 = sp500.history(
    start = f'{start_year}-01-01', 
    end = '2022-01-01', 
    interval=interval_dict.get(interval))

st.line_chart(sp500['Close'])

st.markdown('---')



# market = yf.Ticker(markets_dict.get(index))
# market = market.history(start=start_year, interval=interval)


#plot_Data = market.plot.line(y="Close", use_index=True)

# st.dataframe(market)


# col1, col2 = st.columns(2)
# col1.write("This is column 1")
# col2.write("This is column 2")

