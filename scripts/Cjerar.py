from pandas.tseries.offsets import Hour
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit.elements.arrow import Data

def write():
  st.title("Cluster Jerarquico")
  uploaded_file = st.file_uploader("Elige un CSV")
  if uploaded_file is not None:
    # To read file as bytes:
    dataframe = pd.read_csv(uploaded_file)
    st.write(dataframe)
    
    columns_names = dataframe.columns.values

    st.markdown('A continuacion elige la variable por la que se va a agrupar')

    predecida = st.selectbox('Variable ...', columns_names)
    # st.write('You selected:', predecida)
    # st.pyplot(sns.pairplot(dataframe, hue=str(predecida)))   DESCOMENTAR

    st.markdown("""
    ## Matriz de correlaciones

    Una matriz de correlaciones es útil para analizar la relación entre las variables numéricas. Se emplea la función corr()

    """)

    CorrHipoteca = dataframe.corr(method='pearson')
    st.write(CorrHipoteca)

    st.markdown("""
        Se muestra la correlacion con el metodo de pearson
    """)
    #Mapa de calor de correlaciones
    fig, ax = plt.subplots()
    MatrizInf = dataframe.corr(method='pearson')
    MatrizInf = np.triu(CorrHipoteca)
    sns.heatmap(CorrHipoteca, cmap='RdBu_r', annot=True, mask=MatrizInf)
    st.pyplot(fig) 

    st.markdown("""
    Lo que nos deja con las siguietes variables suegeridas como predictorias
    """)
    varPredic = CorrHipoteca['ingresos'].sort_values(ascending=False)[:10]
    varPredic = varPredic.drop('ingresos', 0)
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

    columns_namess = MatrizActual.columns.values

    st.markdown("""
    
    Cuando se trabaja con clustering, dado que son algoritmos basados en distancias, 
    es fundamental escalar los datos para que cada una de las variables contribuyan 
    por igual en el análisis.
    """)
    import scipy.cluster.hierarchy as shc
    from sklearn.cluster import AgglomerativeClustering

    MatrizHipoteca = np.array(dataframe[columns_namess])
    from sklearn.preprocessing import StandardScaler, MinMaxScaler  
    estandarizar = StandardScaler()                               # Se instancia el objeto StandardScaler o MinMaxScaler 
    MEstandarizada = estandarizar.fit_transform(MatrizHipoteca)   # Se calculan la media y desviación y se escalan los datos

    #Se crean las etiquetas de los elementos en los clústeres
    MJerarquico = AgglomerativeClustering(n_clusters=7, linkage='complete', affinity='euclidean')
    MJerarquico.fit_predict(MEstandarizada)
    MJerarquico.labels_

    st.markdown("""
    Añade la columna al cluster que pertenece
    """)

    Hipoteca = MatrizActual
    Hipoteca['clusterH'] = MJerarquico.labels_
    st.write(Hipoteca) 

    st.markdown("""
    Cada cluster tiene la siguiente cantidad de elementos
    """)
    #Cantidad de elementos en los clusters
    st.write(Hipoteca.groupby(['clusterH'])['clusterH'].count())
    st.markdown("""
    Finalmente se tienen las medias de clada cluster listas para ser interpretadas
    """)
    CentroidesH = Hipoteca.groupby('clusterH').mean()
    st.write(CentroidesH)
    

