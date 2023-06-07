from pandas.tseries.offsets import Hour
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit.elements.arrow import Data

def write():
    st.title("Árboles de desición (Clasificación)")

    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score
    from sklearn import model_selection

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
      
      dataframe = dataframe.replace({'M': 'Malignant', 'B': 'Benign'})

      st.markdown("""
      De acuerdo al mapa de calor elige las variables predictorias que formaran parte del modelo
      """)
      columns_names = dataframe.columns.values
      # st.write(columns_names)

      predictorias = st.multiselect('Variables....', columns_names)
      # st.write('Haz elegido:', predictorias)
      
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

      from sklearn.tree import DecisionTreeClassifier
      from sklearn.metrics import classification_report
      from sklearn.metrics import confusion_matrix
      from sklearn.metrics import accuracy_score
      from sklearn import model_selection


      X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size = 0.2, random_state = 0, shuffle = True)

      st.markdown("""
      ## Entrenamiento del modelo
      """)

      #Se entrena el modelo
      ClasificacionAD = DecisionTreeClassifier()
      ClasificacionAD.fit(X_train, Y_train)

      
      #Se genera el pronostico
      Y_Clasificacion = ClasificacionAD.predict(X_validation)
      st.write(pd.DataFrame(Y_Clasificacion))

      st.markdown("""
      ## Genera Clasificaciones
      """)
      Valores = pd.DataFrame(Y_validation, Y_Clasificacion)
      st.write(Valores)

      st.markdown("""
      ## Parametros del modelo
      """)

      st.write('Matriz de clasificación')
      Y_Clasificacion = ClasificacionAD.predict(X_validation)
      Matriz_Clasificacion = pd.crosstab(Y_validation.ravel(), 
                                        Y_Clasificacion, 
                                        rownames=['Real'], 
                                        colnames=['Clasificación']) 
      st.write(Matriz_Clasificacion)

      st.write("Criterio: " + ClasificacionAD.criterion)
      st.write('**Exactitud: **' + str(ClasificacionAD.score(X_validation, Y_validation)))
      st.write(classification_report(Y_validation, Y_Clasificacion))
      
      Importancia = pd.DataFrame({'Variable': list(dataframe[predictorias]),
                            'Importancia': ClasificacionAD.feature_importances_}).sort_values('Importancia', ascending=False)
      st.write(Importancia)

      
      from sklearn.tree import export_text
      Reporte = export_text(ClasificacionAD, feature_names = predictorias)
      st.write(Reporte)

                                      