from pandas.tseries.offsets import Hour
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit.elements.arrow import Data

def write():
    st.title("Arboles de desición (Pronostico)")

    uploaded_file = st.file_uploader("Elige un CSV")
    if uploaded_file is not None:
      # To read file as bytes:
      dataframe = pd.read_csv(uploaded_file)
      st.write(dataframe)

      st.markdown("""
      A continuación se muestran los principales indicadores estadisticos del dataframe
      """)
      st.write(dataframe.describe())

      st.markdown("""
      A continuación se muestra un mapa de calor donde se pueden ver la correlación
      """)
      #Mapa de calor de correlaciones
      fig, ax = plt.subplots()
      MatrizInf = np.triu(dataframe.corr())
      sns.heatmap(dataframe.corr(), cmap='RdBu_r', annot=True, mask=MatrizInf)
      st.write(fig)

      st.markdown("""
      De acuerdo al mapa de calor elige las variables predictorias que formaran parte del modelo
      """)
      columns_names = dataframe.columns.values
      # st.write(columns_names)

      predictorias = st.multiselect('Variables....', columns_names)
      # st.write('Haz elegido:', predictorias)

      from sklearn.tree import DecisionTreeRegressor
      from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
      from sklearn import model_selection


      X = np.array(dataframe[predictorias])
      # pd.DataFrame(X)
      st.write(pd.DataFrame(X))

      st.markdown("""
      De acuerdo al mapa de calor elige la variable a pronosticar que formaran parte del modelo
      """)

      predecida = st.selectbox('Variable ...', columns_names)
      # st.write('You selected:', predecida)
      Y = np.array(dataframe[predecida])
      st.write(pd.DataFrame(Y))

      X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size = 0.2, random_state = 1234, shuffle = True)

      st.markdown("""
      ## Entrenamiento del modelo
      """)

      #Se entrena el modelo
      PronosticoAD = DecisionTreeRegressor()
      PronosticoAD.fit(X_train, Y_train)

      
      #Se genera el pronostico
      Y_Pronostico = PronosticoAD.predict(X_test)
      st.write(pd.DataFrame(Y_Pronostico))

      st.markdown("""
      ## Genera Pronostico
      """)
      Valores = pd.DataFrame(Y_test, Y_Pronostico)
      st.write(Valores)

      st.markdown("""
      ## Parametros del modelo
      """)

      st.write("Criterio: " + PronosticoAD.criterion)
      st.write('MAE: ' + str(mean_absolute_error(Y_test, Y_Pronostico)))
      st.write('MSE: ' + str(mean_squared_error(Y_test, Y_Pronostico)))
      st.write('RMSE: ' + str(mean_squared_error(Y_test, Y_Pronostico, squared=False)))
      st.write('Score: ' + str(r2_score(Y_test, Y_Pronostico)))
      
      Importancia = pd.DataFrame({'Variable': list(dataframe[predictorias]),
                            'Importancia': PronosticoAD.feature_importances_}).sort_values('Importancia', ascending=False)
      st.write(Importancia)

      import graphviz
      from sklearn.tree import export_graphviz
      # Se crea un objeto para visualizar el árbol
      # Se incluyen los nombres de las variables para imprimirlos en el árbol
      Elementos = export_graphviz(PronosticoAD, feature_names = predictorias)  
      Arbol = graphviz.Source(Elementos)
      st.graphviz_chart(Arbol)

      
      
      
      
