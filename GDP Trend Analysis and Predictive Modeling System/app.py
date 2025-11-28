import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px # type: ignore
from statsmodels.tsa.arima.model import ARIMA # type: ignore
import warnings
warnings.filterwarnings("ignore")

st.title("GDP Statistics Evaluation & Forecasting System")

# Sidebar - Country Input
countries = st.sidebar.text_input(
    "Enter ISO Country Code(s), comma separated:",
    ""
)
country_codes = [code.strip().upper() for code in countries.split(",") if code.strip()]

# Sidebar - Year Range
start_year, end_year = st.sidebar.slider("Select Duration", 1960, 2025, (1970, 2022))

@st.cache_data
def fetch_gdp_data(country_code):
    url = f"http://api.worldbank.org/v2/country/{country_code}/indicator/NY.GDP.MKTP.CD?format=json&per_page=500"
    response = requests.get(url)
    if response.status_code != 200:
        return pd.DataFrame()
    try:
        data = response.json()[1]
    except Exception:
        return pd.DataFrame()

    df = pd.DataFrame([{
        "country": country_code,
        "year": int(item['date']),
        "gdp": item['value']
    } for item in data if item['value'] is not None])

    return df

# Fetch all data
whole_data = pd.DataFrame()
for code in country_codes:
    df_country = fetch_gdp_data(code)
    if df_country.empty:
        st.warning(f"No GDP data available for {code}.")
    else:
        whole_data = pd.concat([whole_data, df_country], ignore_index=True)

if whole_data.empty:
    st.error("No GDP data available for the selected countries.")
    st.stop()

# Filter by year
df_filtered = whole_data[(whole_data['year'] >= start_year) & (whole_data['year'] <= end_year)]

# Dynamic filters
if len(country_codes) > 1:
    selected_country = st.sidebar.multiselect(
        "Filter by Country",
        options=df_filtered['country'].unique(),
        default=df_filtered['country'].unique()
    )
    df_filtered = df_filtered[df_filtered['country'].isin(selected_country)]

    min_year, max_year = int(df_filtered['year'].min()), int(df_filtered['year'].max())
    year_range = st.sidebar.slider("Filter by Year", min_year, max_year, (min_year, max_year))
    df_filtered = df_filtered[(df_filtered['year'] >= year_range[0]) & (df_filtered['year'] <= year_range[1])]

# Show filtered data
st.subheader("GDP Data Table")
st.dataframe(df_filtered.sort_values(by=['country', 'year']))

# Main GDP Line Chart
st.subheader("GDP Analysis Over Time")
fig = px.line(
    df_filtered,
    x='year',
    y='gdp',
    color='country',
    title=f"GDP Comparison ({start_year}–{end_year})",
    labels={'year': 'Year', 'gdp': 'GDP (current US$)', 'country': 'Country'}
)
st.plotly_chart(fig)

# Summary statistics
st.subheader("Summary Statistics by Country")
stats = df_filtered.groupby("country")["gdp"].describe()
st.dataframe(stats)

# ────────────────────────────────────────────────────────────────
#   ADVANCED ARIMA FORECAST MODULE (AUTO MODEL + CI)
# ────────────────────────────────────────────────────────────────

def auto_arima_select(ts):
    """
    Simple auto-ARIMA search for the best (p,d,q) using AIC score.
    Search range kept small so it's FAST for Streamlit.
    """
    best_aic = np.inf
    best_order = None
    best_model = None

    for p in range(0, 4):
        for d in range(0, 2):
            for q in range(0, 4):
                try:
                    model = ARIMA(ts, order=(p, d, q))
                    model_fit = model.fit()
                    if model_fit.aic < best_aic:
                        best_aic = model_fit.aic
                        best_order = (p, d, q)
                        best_model = model_fit
                except:
                    continue

    return best_order, best_model


def forecast_gdp_arima(df_country, forecast_years=10):
    df = df_country.sort_values("year")
    ts = df.set_index("year")["gdp"]

    # Auto ARIMA
    order, model = auto_arima_select(ts)

    # Forecast + CI
    forecast_res = model.get_forecast(steps=forecast_years)
    mean_forecast = forecast_res.predicted_mean
    conf_int = forecast_res.conf_int()

    future_years = np.arange(df['year'].max() + 1, df['year'].max() + forecast_years + 1)

    df_forecast = pd.DataFrame({
        "year": future_years,
        "forecast_gdp": mean_forecast.values,
        "lower_ci": conf_int.iloc[:, 0].values,
        "upper_ci": conf_int.iloc[:, 1].values
    })

    return df_forecast, order


# ────────────────────────────────────────────────────────────────
#   DISPLAY FORECAST SECTION
# ────────────────────────────────────────────────────────────────

st.subheader("GDP Forecast for next 10 years")

unique_countries = df_filtered['country'].unique()

for country in unique_countries:
    st.write(f"### {country} Forecast")

    df_country = whole_data[whole_data["country"] == country]

    df_forecast, order = forecast_gdp_arima(df_country)

    #st.write(f"**Selected ARIMA Order:** {order}")

    st.dataframe(df_forecast)

    # Plot with CI
    fig_f = px.line(
        df_forecast,
        x="year",
        y="forecast_gdp",
        title=f"{country} 10-Year GDP Forecast",
    )

    fig_f.add_scatter(
        x=df_forecast["year"],
        y=df_forecast["lower_ci"],
        mode="lines",
        name="Lower CI",
        line=dict(dash="dot")
    )

    fig_f.add_scatter(
        x=df_forecast["year"],
        y=df_forecast["upper_ci"],
        mode="lines",
        name="Upper CI",
        line=dict(dash="dot")
    )

    st.plotly_chart(fig_f)
