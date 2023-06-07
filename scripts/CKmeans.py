from pandas.core.frame import DataFrame
from pandas.tseries.offsets import Hour
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit.elements.arrow import Data

def write():
    st.title("Clusterizacion por K Means")

    uploaded_file = st.file_uploader("Elige un CSV")
    if uploaded_file is not None:
      # To read file as bytes:
      dataframe = pd.read_csv(uploaded_file)
      st.write(dataframe)

      columns_names = dataframe.columns.values

      st.markdown('A continuacion elige la variable por la que se va a agrupar')

      predecida = st.selectbox('Variable ...', columns_names)
      # st.write('You selected:', predecida)
      st.pyplot(sns.pairplot(dataframe, hue=str(predecida)))

      st.markdown("""
        Se muestra la correlacion con el metodo de pearson
      """)
      #Mapa de calor de correlaciones
      fig, ax = plt.subplots()
      CorrBCancer = dataframe.corr(method='pearson')
      MatrizInf = np.triu(CorrBCancer)
      sns.heatmap(CorrBCancer, cmap='RdBu_r', annot=True, mask=MatrizInf)
      st.pyplot(fig)  
      st.markdown("""
      Lo que nos deja con las siguietes variables suegeridas como predictorias
      """)
      varPredic = CorrBCancer['Radius'].sort_values(ascending=False)[:10]
      varPredic = varPredic.drop('Radius', 0)
      st.write(varPredic)
      
      st.markdown("""
      A continuacion selecciona las variables que no formaran parte del modelo
      """)  

      eliminadas = st.multiselect('Variables....', columns_names)

      st.markdown("""
      Lo que nos deja con la siguiente matriz
      """)  

      MatrizActual = dataframe.drop(eliminadas, axis=1)
      st.write(MatrizActual)

      MatrizVariables = np.array(MatrizActual[MatrizActual.columns.values])

      #Se importan las bibliotecas
      from sklearn.cluster import KMeans
      from sklearn.metrics import pairwise_distances_argmin_min

      #Definición de k clusters para K-means
      #Se utiliza random_state para inicializar el generador interno de números aleatorios
      SSE = []
      for i in range(2, 12):
          km = KMeans(n_clusters=i, random_state=0)
          km.fit(MatrizVariables)
          SSE.append(km.inertia_)
      
      st.markdown("""
        Ahora se obtiene el numero de clusters que se recomienda
      """)

      from kneed import KneeLocator
      kl = KneeLocator(range(2, 12), SSE, curve="convex", direction="decreasing")
      st.write(kl.elbow)
      
      #Se crean las etiquetas de los elementos en los clusters
      MParticional = KMeans(n_clusters=5, random_state=0).fit(MatrizVariables)
      MParticional.predict(MatrizVariables)
      MParticional.labels_

      dataframe['clusterP'] = MParticional.labels_
      # st.write(dataframe)

      st.markdown("""
        Una vez obtenido el numero de clusters se realiza la agrupacion y se obtiene el 
        tamaño de los clusters
      """)

      st.write(dataframe.groupby(['clusterP'])['clusterP'].count())

      CentroidesP = MParticional.cluster_centers_
      st.write(pd.DataFrame(CentroidesP.round(4), columns=[MatrizActual.columns.values]))
      
      st.markdown("""
        Se presenta una grafica en 3D de los elementos y centroides de los clusters
      """)

      # Gráfica de los elementos y los centros de los clusters
      from mpl_toolkits.mplot3d import Axes3D
      plt.rcParams['figure.figsize'] = (10, 7)
      plt.style.use('ggplot')
      colores=['red', 'blue', 'cyan', 'green', 'yellow']
      asignar=[]
      for row in MParticional.labels_:
          asignar.append(colores[row])

      fig = plt.figure()
      ax = Axes3D(fig)
      ax.scatter(MatrizVariables[:, 0], MatrizVariables[:, 1], MatrizVariables[:, 2], marker='o', c=asignar, s=60)
      ax.scatter(CentroidesP[:, 0], CentroidesP[:, 1], CentroidesP[:, 2], marker='o', c=colores, s=1000)
      st.pyplot(fig)

      st.markdown("""
      Es posible identificar los casos más cercanos a cada centroide
      """)
      #Es posible identificar los pacientes más cercanos a cada centroide
      Cercanos,_ = pairwise_distances_argmin_min(MParticional.cluster_centers_, MatrizVariables)
      Pacientes = dataframe['IDNumber'].values
      for row in Cercanos:
          st.write(Pacientes[row])