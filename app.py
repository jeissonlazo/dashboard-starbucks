import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression

# -------------------------

# CONFIG

# -------------------------

st.set_page_config(page_title="Dashboard Starbucks", layout="wide")

st.title("☕ Dashboard de Análisis de Clientes Starbucks")

# -------------------------

# LOAD DATA

# -------------------------

df = pd.read_csv("data/starbucks_customers.csv")

# -------------------------

# FILTRO (INTERACTIVO)

# -------------------------

st.sidebar.header("Filtros")

channel = st.sidebar.selectbox(
"Selecciona canal",
df["order_channel"].unique()
)

filtered_df = df[df["order_channel"] == channel]

# -------------------------

# GRÁFICAS

# -------------------------

st.subheader("Distribución del gasto")
fig_hist = px.histogram(filtered_df, x="total_spend", nbins=50)
st.plotly_chart(fig_hist, use_container_width=True)

st.subheader("Gasto por canal")
fig_box = px.box(df, x="order_channel", y="total_spend")
st.plotly_chart(fig_box, use_container_width=True)

st.subheader("Relación cart_size vs total_spend")
fig_scatter = px.scatter(filtered_df, x="cart_size", y="total_spend")
st.plotly_chart(fig_scatter, use_container_width=True)

# -------------------------

# MODELO

# -------------------------

st.subheader("Modelo de regresión")

X = df[["cart_size", "num_customizations", "fulfillment_time_min"]]
y = df["total_spend"]

model = LinearRegression()
model.fit(X, y)

df["prediction"] = model.predict(X)

fig_pred = px.scatter(df, x="total_spend", y="prediction",
title="Real vs Predicción")

st.plotly_chart(fig_pred, use_container_width=True)

# -------------------------

# MÉTRICAS

# -------------------------

from sklearn.metrics import r2_score, mean_squared_error

r2 = r2_score(y, df["prediction"])
mse = mean_squared_error(y, df["prediction"])

col1, col2 = st.columns(2)

col1.metric("R²", f"{r2:.3f}")
col2.metric("MSE", f"{mse:.3f}")

# -------------------------

# CONCLUSIONES

# -------------------------

st.subheader("Conclusiones")

st.write("""

* El canal influye en el gasto de los clientes.
* Mobile App presenta mayor gasto promedio.
* El modelo tiene buen desempeño predictivo.
  """)
