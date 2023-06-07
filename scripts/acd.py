import streamlit as st
import pandas as pd
from pandas.tseries.offsets import Hour
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit.elements.arrow import Data


def write():
    st.title('Seleccion De Caracteristicas')
    def load_data(nrows):
        data = pd.read_csv("inci.csv", index_col=0, encoding='latin-1', nrows=nrows)
        return data

    # Create a text element and let the reader know the data is loading.
    data_load_state = st.text('Cargando datos...')
    # Load 10,000 rows of data into the dataframe.
    # maximo numero 217000
    data = load_data(10000)
    # Notify the reader that the data was successfully loaded.
    data_load_state.text("Datos leidos con exito!")

    # st.subheader('Raw data')
    # st.write(data)
    if st.checkbox('Ver datos'):
        st.subheader('Datos...')
        st.write(data)
    st.markdown("## Evaluación Visual")
    st.pyplot(sns.pairplot(data, hue='delegacion_inicio'))

    #Mapa de calor de correlaciones
    fig, ax = plt.subplots()
    sns.heatmap(data.corr(), ax=ax)
    st.write(fig)

    st.markdown("""
      Tras un recorrido visual y despues de obtener el mapa de calor se van a eliminar algunas columnas que no tiene nada que ver
    """)
    Data2 = data.drop(['incidente_c4','folio','fecha_cierre','hora_cierre','ano_cierre','delegacion_cierre','clas_con_f_alarma','tipo_entrada','mes_cierre'], axis=1)
    st.write(Data2)
    Data2.to_csv('incid.csv')
    