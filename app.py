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
st.markdown(
    """
    ### Análisis de la distribución del gasto total

    La gráfica muestra la distribución del gasto total (`total_spend`) registrado en las transacciones. 
    Se utilizó un histograma para observar la frecuencia de los valores y la forma general de la distribución.

    Se observa una **asimetría positiva (sesgo hacia la derecha)**. La mayor concentración de transacciones 
    se encuentra entre **10 y 18 unidades monetarias**, con un pico cercano a **12–14**, lo que indica que este 
    es el rango de gasto más común.

    También existe una **cola hacia valores altos**, llegando hasta aproximadamente **40 unidades**, lo que indica 
    que hay un grupo pequeño de clientes con gastos significativamente mayores.

    Este comportamiento sugiere que la mayoría de los clientes realizan pedidos simples, mientras que algunos 
    realizan compras más grandes o con más personalizaciones.

    Este análisis es importante porque permite entender el comportamiento general del gasto antes de aplicar 
    modelos estadísticos o contrastes de hipótesis.
    
    El análisis de esta distribución es relevante para el estudio, ya que permite comprender el comportamiento general de la variable de interés antes de realizar el contraste de hipótesis y los modelos de regresión. Además, confirma que la variable total_spend presenta suficiente variabilidad para analizar posibles diferencias entre los distintos canales de pedido.
  """
)

st.subheader("Gasto por canal")
fig_box = px.box(df, x="order_channel", y="total_spend")
st.plotly_chart(fig_box, use_container_width=True)

st.markdown(
    """
    ### Análisis del gasto total según el canal de pedido
    
    La Figura X presenta un diagrama de cajas (boxplot) que muestra la distribución del gasto total (total_spend) según el canal de pedido utilizado por los clientes. Los canales considerados en el análisis incluyen Drive-Thru, Mobile App, Kiosk y In-Store Cashier.
    
    El diagrama de cajas permite comparar la mediana, la dispersión de los datos y la presencia de valores atípicos entre los distintos canales de pedido. A partir de la gráfica se observa que el canal Mobile App presenta una mediana de gasto significativamente mayor en comparación con los demás canales. Esto sugiere que los clientes que realizan pedidos a través de la aplicación móvil tienden a gastar más por transacción.

    Por otro lado, los canales Drive-Thru, Kiosk y In-Store Cashier muestran distribuciones de gasto relativamente similares, con medianas más bajas y una menor dispersión en comparación con el canal móvil. Aunque en todos los canales se observan valores atípicos correspondientes a pedidos de mayor valor, estos aparecen con mayor frecuencia en los pedidos realizados mediante la aplicación móvil.
    
    Estos resultados preliminares sugieren que el canal de pedido podría influir en el comportamiento de gasto de los clientes. En particular, el uso de canales digitales como la aplicación móvil parece estar asociado con pedidos de mayor valor. Este comportamiento podría explicarse por factores como una mayor facilidad para personalizar los productos, promociones digitales o una mayor planificación del pedido por parte del cliente.
    
    Sin embargo, aunque el análisis visual sugiere diferencias entre los canales, es necesario realizar un contraste de hipótesis formal para determinar si estas diferencias son estadísticamente significativas o si podrían explicarse por variaciones aleatorias en los datos. En la siguiente sección se aplicará una prueba t de Student para comparar las medias de gasto entre los canales digitales y los canales tradicionales.
  """
)

st.subheader("Relación cart_size vs total_spend")
fig_scatter = px.scatter(filtered_df, x="cart_size", y="total_spend")
st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown(
    """
    ### Relación entre el tamaño del carrito y el gasto total
    
    La Figura X muestra la relación entre el tamaño del carrito de compra (cart_size) y el gasto total (total_spend) de los clientes mediante un diagrama de dispersión. Cada punto representa una transacción individual dentro del conjunto de datos.
    A partir de la visualización se observa una tendencia positiva clara entre ambas variables. A medida que aumenta el número de productos incluidos en el carrito de compra, el gasto total de la transacción también tiende a incrementarse. Por ejemplo, los pedidos con uno o dos productos generalmente presentan niveles de gasto más bajos, mientras que los pedidos con carritos más grandes (entre 7 y 10 productos) muestran valores de gasto considerablemente mayores.

    Además, se puede notar que, aunque existe cierta variabilidad en los valores de gasto para un mismo tamaño de carrito, la tendencia general mantiene un patrón creciente relativamente consistente. Esto sugiere que el tamaño del carrito es un factor relevante para explicar el gasto total de los clientes.

    Este comportamiento es consistente con la lógica del consumo en retail, donde cada producto adicional agregado al carrito contribuye al incremento del monto final de la compra. Por esta razón, la variable cart_size podría ser considerada un predictor importante en modelos de regresión destinados a estimar o predecir el gasto total de los clientes.

    En análisis posteriores, esta relación podría explorarse mediante un modelo de regresión lineal, con el objetivo de cuantificar la fuerza de la relación entre ambas variables y evaluar qué tan bien el tamaño del carrito explica la variabilidad del gasto total.


    """
)

# -------------------------

# MODELO

# -------------------------


X = df[["cart_size", "num_customizations", "fulfillment_time_min"]]
y = df["total_spend"]

model = LinearRegression()
model.fit(X, y)

df["prediction"] = model.predict(X)

fig_pred = px.scatter(df, x="total_spend", y="prediction",
title="Real vs Predicción")


st.subheader("Modelo de regresión lineal múltiple")

fig_pred = px.scatter(
    df,
    x="total_spend",
    y="prediction",
    labels={"total_spend": "Valor real", "prediction": "Predicción"},
    title="Valores reales vs predichos",
)

# Línea ideal (y = x)
fig_pred.add_shape(
    type="line",
    x0=df["total_spend"].min(),
    y0=df["total_spend"].min(),
    x1=df["total_spend"].max(),
    y1=df["total_spend"].max(),
    line=dict(dash="dash"),
)

st.plotly_chart(fig_pred, use_container_width=True)

st.markdown(
    """ 
            ### Modelo de regresión lineal múltiple
            
            Con el objetivo de analizar cómo diferentes factores influyen en el gasto total de los clientes, se implementó un modelo de regresión lineal múltiple. Este tipo de modelo permite estudiar la relación entre varias variables independientes (características) y una variable dependiente.
            
            En este caso, se seleccionó como variable dependiente el gasto total del pedido (`total_spend`), mientras que las variables independientes fueron:
            
            - `cart_size`: número de productos en el carrito.
            - `num_customizations`: número de personalizaciones realizadas en el pedido.
            - `fulfillment_time_min`: tiempo de preparación o entrega del pedido en minutos.
            
            Estas variables fueron elegidas porque representan factores relevantes del comportamiento de compra del cliente y podrían influir en el monto total gastado en cada transacción.

            Para entrenar el modelo, el conjunto de datos se dividió en dos subconjuntos utilizando la función train_test_split de la biblioteca scikit-learn. Se empleó una división 70%-30%, donde el 70% de los datos se utilizó para entrenar el modelo 
            
            y el 30% restante para evaluar su desempeño. Esta proporción es común en problemas de aprendizaje automático ya que permite contar con suficientes datos para el entrenamiento sin comprometer la capacidad de evaluar el modelo con datos no vistos previamente.

            El modelo fue entrenado utilizando la clase LinearRegression de scikit-learn y posteriormente se realizaron predicciones sobre el conjunto de prueba utilizando el método predict

            Los coeficientes de regresión obtenidos mediante el atributo coef_ indican el impacto que tiene cada variable independiente sobre el gasto total. En particular, el coeficiente asociado a cart_size refleja cuánto aumenta el gasto total cuando se agrega un producto adicional al carrito. De manera similar, el coeficiente de num_customizations muestra el efecto de las personalizaciones en el gasto final del cliente, mientras que el coeficiente de fulfillment_time_min permite evaluar si existe alguna relación entre el tiempo de preparación del pedido y el monto gastado.

            El modelo también incluye un término de intercepción (intercept_), que representa el valor esperado del gasto total cuando todas las variables independientes toman el valor cero. Este término forma parte de la ecuación de regresión y contribuye a ajustar correctamente el modelo a los datos observados.


            Para evaluar el desempeño del modelo se calcularon dos métricas principales: el Error Cuadrático Medio (MSE) y el coeficiente de determinación (R²). El MSE mide el promedio de los errores al cuadrado entre los valores predichos y los valores reales, mientras que el R² indica qué proporción de la variabilidad del gasto total puede ser explicada por las variables incluidas en el modelo.


            En conjunto, estos resultados permiten evaluar qué tan bien el modelo logra capturar la relación entre las características seleccionadas y el gasto total de los clientes.

            """
)

coef_df = pd.DataFrame(
    {
        "Variable": ["cart_size", "num_customizations", "fulfillment_time_min"],
        "Coeficiente": model.coef_,
    }
)

fig_coef = px.bar(
    coef_df,
    x="Variable",
    y="Coeficiente",
    title="Importancia de variables en el modelo",
)

st.plotly_chart(fig_coef, use_container_width=True)


st.subheader("Mapa de correlaciones")

# Seleccionar variables numéricas
numeric_df = df.select_dtypes(include=["number"])

# Calcular correlación
corr = numeric_df.corr()

# Crear heatmap
fig_corr = px.imshow(
    corr,
    text_auto=True,
    aspect="auto",
    color_continuous_scale="RdBu_r",
    title="Matriz de correlación",
)

st.plotly_chart(fig_corr, use_container_width=True)

st.markdown(
    """
            ###Contraste de hipótesis mediante prueba t de Student
            
            - `T statistic`: 123.37936121243389
            - `P-value`: 0.0

            Para evaluar si existe una diferencia significativa en el gasto total de los clientes según el canal de pedido, se aplicó una prueba t de Student para dos muestras independientes. En este análisis se compararon las transacciones realizadas a través de canales digitales (Mobile App) frente a canales tradicionales (Drive-Thru, Kiosk e In-Store Cashier).


            El objetivo de esta prueba es determinar si la diferencia observada en los promedios de gasto entre estos grupos es estadísticamente significativa o si podría deberse al azar.
            Los resultados obtenidos fueron los siguientes:


            - Estadístico t: 123.38
            - p-valor: 0.0
            - Nivel de significancia: α = 0.05
            
            
            El estadístico t mide la magnitud de la diferencia entre las medias de los grupos en relación con la variabilidad de los datos. En este caso, el valor obtenido es muy alto (t = 123.38), lo que indica una diferencia considerable entre los grupos analizados.
            
            
            El p-valor obtenido es aproximadamente 0.0, lo que significa que la probabilidad de observar una diferencia tan grande entre los grupos si la hipótesis nula fuera verdadera es extremadamente baja.

          Dado que:
          
          p-value < α (0.0 < 0.05)
          
          se rechaza la hipótesis nula (H₀).
          
          Por lo tanto, existe evidencia estadísticamente significativa para afirmar que el canal de pedido influye en el gasto total de los clientes, siendo el canal Mobile App el que presenta un mayor gasto promedio por transacción en comparación con los canales tradicionales.


            """
)
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
