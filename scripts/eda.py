import streamlit as st
import pandas as pd
from pandas.tseries.offsets import Hour
import numpy as np
from streamlit.elements.arrow import Data


# pylint: disable=line-too-long
def write():
  
    st.title('Análisis Exploratorio de Datos')
    st.markdown('''---''')

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

    st.markdown('### Descripcion de la estructura de los datos')
    st.markdown('La matriz de datos que se cargo tiene la siguiente forma, cabe mencionar que streamlit no permite cargar mas de 50 MB de datos, po lo que solo se eligen 10000 registros de forma aleatoria')
    st.write(data.shape)

    st.markdown('El tipo de dato que se tiene por columna')
    # st.write(data.dtypes) Pendiente error

    st.markdown('### Identifcación de datos faltantes')
    st.write(data.isnull().sum())
    st.markdown('De acuerdo con lo anterior **no se cuentan con datos nulos**')

    st.markdown('### Deteccion de valores atipicos')
    st.markdown('En la siguiente tabla se muestran los principales indicadores estadisticos')
    st.write(data.describe())
    st.markdown('Ahora se muestra la correlacion de variables')
    st.write(data.corr())

    # Convertimos a hora la columna
    Matriz = np.array(data[['latitud', 'longitud', 'hora_creacion']])
    MatrizPD = pd.DataFrame(Matriz, columns=['lat','lon', 'hora_creacion'])
    MatrizPD['hora_creacion'] = pd.to_datetime(MatrizPD['hora_creacion'], format='%H:%M:%S')

    # Mapa de accidentes por hora
    st.subheader('Mapa de accidentes por hora')
    # st.map(MatrizPD)
    hour_to_filter = st.slider('hour', 0, 23, 17)  # min: 0h, max: 23h, default: 17h
    filtered_data = MatrizPD[MatrizPD['hora_creacion'].dt.hour == hour_to_filter]
    st.subheader(f'Accidentes a las {hour_to_filter}:00')
    # st.write(filtered_data)
    st.map(filtered_data)

    st.subheader('Histograma de accidentes de acuerdo a la hora del dia')
    # Grafica de accidentes por hora
    hist_values = np.histogram(
        MatrizPD['hora_creacion'].dt.hour, bins=24, range=(0,24))[0]

    st.bar_chart(hist_values)