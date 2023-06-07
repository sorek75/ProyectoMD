from os import write
import streamlit as st

import streamlit as ast
import scripts.acercade
import scripts.eda
import scripts.acd
import scripts.pca
import scripts.Cjerar
import scripts.CKmeans
import scripts.apriori
import scripts.Aprono
import scripts.Aclass


PAGES = {
    "Acerca de": scripts.acercade,#R
    "Análsis Exploratorio de Datos": scripts.eda,#R
    "Seleccion de caracteristicas": scripts.acd,#R
    "Analsis de componentes principales": scripts.pca,
    "Cluster jerarquico": scripts.Cjerar,#R
    "Clusterizacion por K-Means": scripts.CKmeans,#R
    "Reglas de asociacion": scripts.apriori,#Falta imprimir grafica
    "Pronostico por arboles": scripts.Aprono,#Falta imprimir el arbol y meter prono
    "Clasificacion por arboles": scripts.Aclass,#Falta imprimir el arbol y meter prono
}

def main():
    """Index"""
    st.sidebar.title("Menú")
    selection = st.sidebar.radio("Seleccionar...", list(PAGES.keys()))

    st.sidebar.title("Contacto")
    st.sidebar.success("""
        **Alumno:** Erick Rodrigo Minero Pineda
        **Correo:** rodreri@gmail.com
    """)

    page = PAGES[selection]

    with st.spinner(f"Cargando {selection} ..."):
        ast.shared.components.write_page(page)


if __name__ == "__main__":
    main()