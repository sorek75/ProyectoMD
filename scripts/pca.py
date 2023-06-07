import streamlit as st
import pandas as pd
from pandas.tseries.offsets import Hour
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit.elements.arrow import Data
import awesome_streamlit as ast
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def write():
    st.title('Analisis de componentes principales')
    uploaded_file = st.file_uploader("Elige un CSV")
    if uploaded_file is not None:
      # To read file as bytes:
      dataframe = pd.read_csv(uploaded_file)
      st.write(dataframe)

      st.markdown("""
      # Procedimiento PCA

      1. Se hace una estandarización de los datos
      2. Se calcula la matriz de covarianzas o correlaciones.
      3. Se calculan los componentes (eigen-vectores) y la varianza (eigen-valores).
      4. Se decide el número de componentes principales.
      5. Se examina la proporción de relevancias –cargas–
      """)

      st.markdown("""
      Para poder normalizar es necesario que selecciones las variables nominales, a continuacion seleccionales
      """)
      columns_names = dataframe.columns.values
      eliminadas = st.multiselect('Variables....', columns_names)
      

      normalizar = StandardScaler()                                         # Se instancia el objeto StandardScaler 
      NuevaMatriz = dataframe.drop(eliminadas, axis=1)    # Se quitan las variables no necesarias (nominales)
      normalizar.fit(NuevaMatriz)                                           # Se calcula la media y desviación para cada variable
      MNormalizada = normalizar.transform(NuevaMatriz) 

      st.markdown("""
      ## Paso 1: Matriz normalizada
      """)
      st.write(pd.DataFrame(MNormalizada, columns=NuevaMatriz.columns))
      st.write(MNormalizada.shape)

      st.markdown("""
      ## Pasos 2 y 3: Se calcula la matriz de covarianzas o correlaciones, y se calculan los componentes (eigen-vectores) y la varianza (eigen-valores)
      """)

      pca = PCA(n_components=None)           # Se instancia el objeto PCA, pca=PCA(n_components=None), pca=PCA(.85)
      pca.fit(MNormalizada)                  # Se obtiene los componentes
      st.write(pca.components_)    

      st.markdown("""
      ## Paso 4: Se decide el número de componentes principales

      Se calcula el porcentaje de relevancia, es decir, entre el 75 y 90% de varianza total.
      Se identifica mediante una gráfica el grupo de componentes con mayor varianza.
      **Se elige las dimensiones cuya varianza sea mayor a 1.**
      """)

      Varianza = pca.explained_variance_ratio_
      st.write('Porporción de varianza:', Varianza)
      st.write('Varianza acumulada: ', sum(Varianza[0:3]))   
      st.write('Con 3 componentes se tiene el 88% de varianza acumulada y con 4 el 93%')

      fig, ax = plt.subplots()
      # Se grafica la varianza acumulada en las nuevas dimensiones
      plt.plot(np.cumsum(pca.explained_variance_ratio_))
      plt.xlabel('Número de componentes')
      plt.ylabel('Varianza acumulada')
      plt.grid()
      st.pyplot(fig)  

      st.markdown("""
      ## Paso 5: Se examina la proporción de relevancias cargas

      La importancia de cada variable se refleja en la magnitud de los valores en los componentes (mayor magnitud es sinónimo de mayor importancia).

      Se revisan los valores absolutos de los componentes principales seleccionados. Cuanto mayor sea el valor absoluto, más importante es esa 
      variable en el componente principal.
      """)

      st.write(pd.DataFrame(abs(pca.components_)))

      CargasComponentes = pd.DataFrame(abs(pca.components_), columns=NuevaMatriz.columns)
      st.write(CargasComponentes)