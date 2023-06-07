from pandas.tseries.offsets import Hour
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit.elements.arrow import Data

def write():
    # Datos Generales
    st.title('Universidad Nacional Autónoma de México')
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("# Facultad de Ingeniería")
    with col2:
        st.text("")
    with col3:
        st.image("https://www.ingenieria.unam.mx/nuestra_facultad/images/institucionales/escudo_fi_color.png")


    st.markdown('**Alumno:** Erick Rodrigo Minero Pineda')
    st.markdown('**Correo:** rodreri@gmail.com')
    st.markdown('**Semestre:** 2022-1')

    st.markdown("""
        En este proyecto si integra en una interfaz grafica la compilacion de las practicas
        realizadas a lo largo del semestre.
        
        Se va a utilizar como datos de entrada, datos abiertos que la CDMX publica, con fecha 
        del 20 de noviembre del 2021. Estos datos contienen un conjunto de datos con los incidentes 
        viales reportados por el Centro de Comando, Control, Cómputo, Comunicaciones y Contacto 
        Ciudadano de la Ciudad de México (C5) desde 2014 y actualizado mensualmente
    """)
    
    st.text('Liga: https://datos.cdmx.gob.mx/dataset/incidentes-viales-c5')
    st.markdown("""
        Este proyecto esta diseñado para este set de datos en particular, sin embargo,
        si se omiten los elementos del mapa y del hist, se tiene una aplicacion capaz 
        de funcionar con cualquier set de datos que se ingrese.
    """)