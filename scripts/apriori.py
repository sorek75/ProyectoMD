from pandas.tseries.offsets import Hour
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit.elements.arrow import Data
from apyori import apriori

def write():
    st.title("Reglas de asociación")

    uploaded_file = st.file_uploader("Elige un CSV")
    if uploaded_file is not None:
      # To read file as bytes:
      dataframe = pd.read_csv(uploaded_file, header=None)
      st.write(dataframe)

      st.markdown("""
      ## Procesando los datos

      ### Exploración

      Antes de ejecutar el algoritmo es recomendable observar la distribución de la frecuencia de los elementos.

      """)
      
      #Se incluyen todas las transacciones en una sola lista
      Transacciones = dataframe.values.reshape(149200).tolist() #-1 significa 'dimensión no conocida'
      #Se crea una matriz (dataframe) usando la lista y se incluye una columna 'Frecuencia'
      ListaM = pd.DataFrame(Transacciones)
      st.write(ListaM)
      ListaM['Frecuencia'] = 0
      st.write(ListaM)

      st.markdown("""
      Se agrupan los elementos
      """)
      #Se agrupa los elementos
      ListaM = ListaM.groupby(by=[0], as_index=False).count().sort_values(by=['Frecuencia'], ascending=True) #Conteo
      ListaM['Porcentaje'] = (ListaM['Frecuencia'] / ListaM['Frecuencia'].sum()) #Porcentaje
      ListaM = ListaM.rename(columns={0 : 'Item'})
      # st.write(ListaM)

      # Grafica
      # fig = plt.subplots()
      # plt.ylabel('Item')
      # plt.xlabel('Frecuencia')
      # plt.barh(ListaM['Item'], width=ListaM['Frecuencia'], color='blue')

      st.markdown('## Preapración')
      MoviesLista = dataframe.stack().groupby(level=0).apply(list).tolist()
      # st.write(MoviesLista)

      MoviesL = []
      for i in range(0, 7459):
        MoviesL.append([str(dataframe.values[i,j]) for j in range(0, 20)])
      # st.write(MoviesL)

      NuevaLista = []
      for item in MoviesL:
        if str(item) != 'nan':
          NuevaLista.append(item)
      


      st.markdown("""
      Se va realizar una configuracion a manera de ejemplo: se tienen
       los sigueintes parametros
      min_support=0.01, min_confidence=0.3, min_lift=2
      """)
      ReglasC1 = apriori(MoviesLista, min_support=0.01, min_confidence=0.3, min_lift=2)
      ResultadosC1 = list(ReglasC1)
      print(len(ResultadosC1)) #Total de reglas encontradas 
      
      for item in ResultadosC1:
        #El primer índice de la lista
        Emparejar = item[0]
        items = [x for x in Emparejar]
        st.write('Regla: ' + str(item[0]))
        # print("Regla: " + str(item[0]))

        #El segundo índice de la lista
        st.write('Soporte ' + str(item[1]))
        # print("Soporte: " + str(item[1]))

        #El tercer índice de la lista
        st.write('Confianza: ' + str(item[2][0][2]))
        # print("Confianza: " + str(item[2][0][2]))
        st.write('Lift: ' + str(item[2][0][3]))
        # print("Lift: " + str(item[2][0][3])) 
        st.write("=====================================")
        # print("=====================================") 
      

      

      