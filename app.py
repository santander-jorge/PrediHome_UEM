
###########################################################################
###########################################################################
#########
######### PREDIHOME
######### URL: https://github.com/santander-jorge/PrediHome_UEM
#########
######### Desarrollado por: 
#########                      Raúl Martín González
#########                      Jorge Santander Trigo
#########                      Aarón William Chacón Morales
#########                      Ricardo Sebastián Quispe Naola
###########################################################################
###########################################################################

from sys import stderr
import pickle
import numpy as np
import streamlit as st
import pandas as pd 
import time 
from streamlit_option_menu import option_menu
import plotly
import plotly.express as px

import webbrowser
import streamlit as st
#from UI import *
import os

from streamlit_keplergl import keplergl_static
from keplergl import KeplerGl

import geopandas as gpd
import json
import streamlit.components.v1 as components
import base64

import replicate

############################ IMPORTANTE: Necesario activar el entorno virtual (venv/Scripts/activate) e instalar las librerías (pip install -r requirements.txt)
# Para ejecutar la aplicación, abrir la terminal y ejecutar el siguiente comando: streamlit run app.py
# Alternativa si este comando no funciona: python -m streamlit run app.py




st.set_page_config(page_title="PrediHome", page_icon="static/images/logo.png", layout="wide")  

# logo_path = "static/images/logo.png"
# st.sidebar.image(logo_path, width=250)

# Añadir un espacio para separar el logo del título
st.write("")
st.markdown("<h1 style='text-align: center; color: orange; font-size: 80px'>PREDIHOME</h1>", unsafe_allow_html=True)



### RUTAS DE LOS ARCHIVOS
ruta_venta_final = './data/Anuncios_venta_dataset_final.csv'
ruta_alquiler_final = './data/Anuncios_alquiler_dataset_final.csv'


    
ruta_medidas_venta = './data/Venta/Distrito_medidas.csv'
ruta_tipovivienda_venta = './data/Venta/Tipo_vivienda.csv'
ruta_rangos_venta = './data/Venta/Anuncio_Venta_ML_Limpio.csv'

ruta_medidas_alquiler = './data/Alquiler/Distrito_medidas_Alquiler.csv'
ruta_tipovivienda_alquiler = './data/Alquiler/Tipo_vivienda_Alquiler.csv'
ruta_rangos_alquiler = './data/Alquiler/Anuncio_Alquiler_ML_Limpio.csv'

ruta_modelo_alquiler = './data/Alquiler/Modelo_Alquiler_ML.pkl'
ruta_modelo_venta = './data/Venta/Modelo_Venta_ML.pkl'



ruta_alquileres_coordenadas = './data/data_web/Alquileres_coordenas_limpio.csv'
ruta_ventas_coordenadas = './data/data_web/Ventas_coordenas_limpio.csv'
ruta_distritos_geo = './data/data_web/Distritos_de_Madrid.geojson'
ruta_municipios_geo = './data/data_web/Municipios.geojson'
ruta_paradas_metro_geo = './data/data_web/PARADAS DE METRO.geojson'
ruta_lineas_metro_geo = './data/data_web/LINEAS.geojson'
ruta_sanidad = './data/data_web/Atención médica.csv'
ruta_bibliotecas = './data/data_web/bibliotecas_madrid.csv'
ruta_polideportivos = './data/data_web/centros_polideportivos.csv'
ruta_colegios = './data/data_web/colegios-publicos_madrid.csv'
ruta_museos = './data/data_web/museos_madrid.csv'
ruta_universidades = './data/data_web/universidades-educacion.csv'


ruta_config_leer = './data/data_web/config_kepler_final.json'
ruta_config_guardar = './data/data_web/config_kepler_final.json'


selected = option_menu(
    menu_title=None,
    options=["Inicio","Inmuebles","Ofertas","Alquila tu piso","Vende tu piso","Mapa","Chatbot"],
    icons=['house','book','book','book','book','map'],
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "15px"}, 
        "nav-link": {"font-size": "15px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "green"},
    }
)

theme_plotly = None

#Declaracion de funciones

#Funcion Home Page
def HomePage():

    file_ = open("static/images/citylarge.gif", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()
    st.markdown(
    f'<img src="data:image/gif;base64,{data_url}" alt="main gif">',
    unsafe_allow_html=True,
    )
         
    # Cargar el archivo CSS
    with open("static/styles/styles.css") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    # Contenido de la aplicación Streamlit
    st.title("Que hacemos?")
    st.title("PrediHome: Análisis de Viviendas")
    st.write("Nuestro proyecto se centra en proporcionar una solución integral para las personas que están interesadas en comprar, vender o alquilar una vivienda. Utilizando técnicas de Machine Learning y análisis de datos geográficos, hemos desarrollado una plataforma que permite a los usuarios obtener predicciones precisas sobre el precio de la vivienda según diversas características.")

    st.write("")
        # Agregar espacio entre las columnas
    st.write("")
    # Columna para las descripciones de lo que ofrecemos
    col1, col2, col3, col4 = st.columns(4)

    # Descripciones de lo que ofrecemos
    with col1:
        st.image("static/img/ml.jpg", caption="Machine Learning para Predicciones Precisas")
        st.markdown("<p style='text-align: center; color: black; font-size: 15px; font-family: Arial, sans-serif;'>Implementamos algoritmos de Machine Learning para analizar grandes conjuntos de datos de viviendas históricas. Estos algoritmos aprenden patrones y relaciones complejas entre diferentes características de las viviendas, como tamaño, número de baños, ubicación geográfica, entre otros. Esto nos permite generar predicciones precisas sobre el precio de compra, venta o alquiler de una vivienda específica.</p>", unsafe_allow_html=True)
        
    with col2:
        st.image("static/img/map.jpg", caption="Localización por Mapa")
        st.markdown("<p style='text-align: center; color: black; font-size: 15px; font-family: Arial, sans-serif;'>Integrando herramientas de análisis geoespacial, nuestra plataforma permite a los usuarios visualizar la ubicación de las viviendas en un mapa interactivo. Esto proporciona una perspectiva única sobre la distribución de las viviendas disponibles en diferentes áreas, así como la proximidad a servicios importantes como escuelas, parques, y centros comerciales.</p>", unsafe_allow_html=True)
        
    with col3:
        st.image("static/img/barra.jpeg", caption="Visualización de Datos con Gráficas")
        st.markdown("<p style='text-align: center; color: black; font-size: 15px; font-family: Arial, sans-serif;'>Utilizamos diversos tipos de gráficas, como barras y de pie, para presentar de manera clara y concisa información sobre las características de las viviendas. Estas visualizaciones permiten a los usuarios comparar fácilmente diferentes atributos, como tamaño, número de habitaciones, y precios, y tomar decisiones informadas.</p>", unsafe_allow_html=True)
        
    with col4:
        st.image("static/img/alvm.jpeg", caption="Ahorro de Dinero y Tiempo")
        st.markdown("<p style='text-align: center; color: black; font-size: 15px; font-family: Arial, sans-serif;'>Nuestra plataforma proporciona una estimación precisa del precio de una vivienda en función de los criterios de búsqueda especificados por el usuario. Esto ayuda a los compradores a evitar sobreprecios y a los vendedores a fijar precios competitivos. Además, al ofrecer una amplia gama de viviendas disponibles para compra, venta o alquiler, los usuarios pueden encontrar rápidamente opciones que se ajusten a sus necesidades y presupuesto, lo que ahorra tiempo y esfuerzo en la búsqueda.</p>", unsafe_allow_html=True)
    
    # Presentación del equipo
    st.title("Quiénes somos?")
    st.write("Somos un equipo interdisciplinario compuesto por cuatro personas con habilidades complementarias:")

    # Columna para las descripciones de lo que ofrecemos
    col1, col2, col3, col4 = st.columns(4)

    # Imágenes de los miembros del equipo con descripciones
    with col1:
        st.image("static/img/foto_sebas.jpeg", caption="Ingeniero")
        st.markdown("<p style='text-align: center; color: black; font-size: 17px; font-family: Open Sans, sans-serif;'>Sebastian Quispe: Encargado del diseño, desarrollo técnico del proyecto y conocimiento en Machine Learning.</p>", unsafe_allow_html=True)
    with col2:
        st.image("static/img/foto_aaron.jpg", caption="Programador")
        st.markdown("<p style='text-align: center; color: black; font-size: 17px; font-family: Arial, sans-serif;'>Aarón Chacón: Responsable de implementar los algoritmos de Machine Learning y la infraestructura tecnológica.</p>", unsafe_allow_html=True)
    with col3:
        st.image("static/img/foto_jorge.jpeg", caption="Sociólogo")
        st.markdown("<p style='text-align: center; color: black; font-size: 17px; font-family: Arial, sans-serif;'>Jorge Santander: Aporta su conocimiento en análisis de datos sociales, crea algoritmos de Machine Learning y tendencias de mercado.</p>", unsafe_allow_html=True)
    with col4:
        st.image("static/img/foto_raul.jpeg", caption="Ingeniero Mecánico")
        st.markdown("<p style='text-align: center; color: black; font-size: 17px; font-family: Arial, sans-serif;'>Raúl Martin: Contribuye con su experiencia en la gestión de proyectos, resolución de problemas complejos y búsqueda de datos.</p>", unsafe_allow_html=True)

    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.title("Sabias que...")
    st.write("")
    st.write("")

    # Pedir al usuario que ingrese la URL
    url = 'https://app.powerbi.com/reportEmbed?reportId=18df8199-26b1-4c98-a6f0-8c90b371cf00&autoAuth=true&ctid=032115c7-35fe-4637-b2c3-d0a42906ba7b'

    # Verificar si se ha ingresado una URL
    if url:
        # Crear el iframe con la URL proporcionada
        iframe_code = f'<iframe src="{url}" width="100%" height="600" justify-content: center style="border:none;"></iframe>'
        # Mostrar el iframe en la aplicación Streamlit
        st.components.v1.html(iframe_code, width=1200, height=600 )

#Def Inmuebles
def Inmuebles():
    try: 
        st.sidebar.header("Por favor filtre aquí:")
        but_alq, but_com = st.sidebar.columns(2)

        Accion = ['Alquilar', 'Vender']

        filtro_accion= st.sidebar.selectbox('Elija una opcion:', Accion, index=None)

        if filtro_accion == 'Alquilar':
            # Importo los datos de alquiler
            df= pd.read_csv(ruta_alquiler_final, sep=';', header=0)
        if filtro_accion == 'Vender':
            # Importo los datos de compras.
            df= pd.read_csv(ruta_venta_final, sep=';', header=0)

        df_vivienda_unique = df['Tipo_vivienda'].unique()
        df_habitaciones_unique = df['Habitaciones'].sort_values(ascending=True).unique()
        df_banos_unique = df['Baños'].sort_values(ascending=True).unique()
        df_Distrito_unique = df['Distrito'].unique()

        Tipo_vivienda = st.sidebar.multiselect('Tipo de vivienda:',df_vivienda_unique,default=df_vivienda_unique[0])
        Distrito = st.sidebar.multiselect('Distrito:',df_Distrito_unique,default=df_Distrito_unique[0])
        Habitaciones = st.sidebar.multiselect('Habitaciones',df_habitaciones_unique, default=df_habitaciones_unique[0])
        Baños = st.sidebar.multiselect('Baños',df_banos_unique, default=df_banos_unique[0])
        df_selection = df.query(
            "Tipo_vivienda == @Tipo_vivienda & Distrito == @Distrito & Habitaciones == @Habitaciones & Baños == @Baños "
        )

        #1. print dataframe
        with st.expander("💵 Mi database de Viviendas en Madrid 🏠", expanded=True):
        #st.dataframe(df_selection,use_container_width=True)
            col_def=[["Tipo_vivienda","Distrito", "Habitaciones","Baños"]]
            shwdata = st.multiselect('Filtro :', df.columns, default=["Tipo_vivienda","Distrito", "Habitaciones","Baños"])
            st.dataframe(df_selection[shwdata],use_container_width=True)

        total_tipo_vivienda = int(df_selection["Tipo_vivienda"].count())
        average_price = round(df_selection.groupby('Tipo_vivienda')["Precio"].mean(), 2)
        #star_rating = ":star:" * int(round(average_price, 0))
        count_tipo_vivienda_distrito = round(df_selection.groupby('Tipo_vivienda')['Precio'].nlargest(5), 2)
        
        #1. simple bar graph
        Tipo_vivienda_Distrito = (
            df_selection.groupby('Distrito')['Tipo_vivienda'].value_counts().groupby(level=0, group_keys=False).nlargest(5).reset_index(name='count')
        )


        fig_Tipo_vivienda_Distrito = px.bar(
            Tipo_vivienda_Distrito,
            x='Distrito',
            y='count',
            color='Tipo_vivienda',
            barmode='group', 
            title='Top 5 Tipos de vivienda por distrito'
            )
        

        fig_Tipo_vivienda_Distrito.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=(dict(showgrid=False)),
            template="plotly_white",
        )
        
        #2. simple line graph------------------
        Barrios_Superficie = df_selection.groupby('Barrio')['Superficie construida'].mean().nlargest(5)
        fig_state = px.bar(
            Barrios_Superficie,
            x=Barrios_Superficie.index,
            #orientation="h",
            y="Superficie construida",
            title="Top 5 Barrios con la mayor media de superficie construida ",
            template="plotly_white",
            color=Barrios_Superficie.index
        )
        fig_state.update_layout(    
            xaxis=dict(tickmode="linear"),
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=(dict(showgrid=False)),
        )

        left_column, right_column,center = st.columns(3)
        left_column.plotly_chart(fig_state, use_container_width=True)
        right_column.plotly_chart(fig_Tipo_vivienda_Distrito, use_container_width=True)

        #pie chart
        with center:
            #fig = px.pie(df_selection, values='Precio', names='Distrito', title='Distrito por Precio')
            #fig.update_layout(legend_title="Distrito", legend_y=0.9)
            #fig.update_traces(textinfo='percent+label', textposition='inside')
            #st.plotly_chart(fig, use_container_width=True, theme=theme_plotly)

            
            fig = px.box(df_selection, x='Precio', y='Distrito', title='<b>Distribución del precio por distrito</b>', color='Distrito', orientation='h')

            st.plotly_chart(fig, use_container_width=True, theme=theme_plotly)
    except:
            st.title('Empiece escogiendo alquiler o venta')
   

#Funcion venta
def ofertas():
    import streamlit as st
    but_alq, but_com = st.sidebar.columns(2)

    Accion = ['Alquilar', 'Vender']

    filtro_accion= st.sidebar.selectbox('Elegir Acción:', Accion, index=None)

    if filtro_accion == None:
        st.title('SELECIONE PRIMERO QUE DESEA ALQUILAR O COMPRAR')

    if filtro_accion == 'Alquilar':
        # Importo los datos de alquiler
        df= pd.read_csv(ruta_alquiler_final, sep=';', header=0)
    if filtro_accion == 'Vender':
        # Importo los datos de compras.
        df= pd.read_csv(ruta_venta_final, sep=';', header=0)


    # Defino la función que muestra las imagenes.
    def main():
        try:
        # Lista de URLs de imágenes
            image_urls= list(df['Url_foto'][(df['Precio'] >= filtro_Precio[0]) & (df['Precio'] <= filtro_Precio[1]) &
            (df['Tipo_vivienda'] == filtro_Tipo_vivienda) & (df['Distrito'] == filtro_Distrito)])
            # ['https://fotos.imghs.net/apps/1018/424/1018_102495424_1_2024013120011114165.jpg',
            #        'https://fotos.imghs.net/apps/2805/30074966438.280500/2805_30074966438_1_20230128125230159.jpg']

            datos = df[['Superficie construida', 'Habitaciones', 'Baños', 'Precio', 'Url_vivienda', 'Descripción']][(df['Precio'] >= filtro_Precio[0]) & (df['Precio'] <= filtro_Precio[1]) &
            (df['Tipo_vivienda'] == filtro_Tipo_vivienda) & (df['Distrito'] == filtro_Distrito)]

            datos.reset_index(inplace=True, drop=True)
            # st.write(datos)

            # Indicador de la imagen actual
            
            current_image_index = st.session_state.get('current_image_index', 0)
        

        # Mostrar la imagen actual
        
            #caption='Imagen desde URL',
            st.image(image_urls[current_image_index], use_column_width=True)
            
            # Botones para cambiar la imagen hacia adelante y hacia atrás
            col1, col2, col3 = st.columns([1, 1, 1])
            if col1.button("◄ Anterior"):
                st.session_state.current_image_index = (current_image_index - 1) % len(image_urls)
                
            if col3.button("Siguiente ►"):
                st.session_state.current_image_index = (current_image_index + 1) % len(image_urls)

            col2.write(f'{current_image_index+1} de {len(image_urls)}')

            st.write(f"Superficie construida: {datos['Superficie construida'][current_image_index]} m², Habitaciones: {datos['Habitaciones'][current_image_index]},    Baños: {datos['Baños'][current_image_index]},    Precio: {datos['Precio'][current_image_index]} €")

            st.write(f"Url_anuncio: /n/n {datos['Url_vivienda'][current_image_index]}")
            
            st.write(f"Descripción: /n/n {datos['Descripción'][current_image_index]}")

        except:
            st.title('NO HAY VIVIENDAS QUE COINCIDAN CON LOS FILTROS APLICADOS')
        
    if filtro_accion != None: 
    # Escribo el título de la ventana y una pequeña explicación de como funciona.
        if filtro_accion == 'Vender':
            st.title('Viviendas en venta')
            min_precio = 20000
            max_precio = 12000000
        if filtro_accion == 'Alquilar':
            st.title('Viviendas en alquiler')
            min_precio = 0
            max_precio = 25000

        st.write('Seleccione los filtros deseados para que le muestre aquellas viviendas que cumplen dichos filtros.')

        import streamlit as st

        st.sidebar.header('Filtros')

        filtro_Tipo_vivienda= st.sidebar.selectbox('Filtrar por tipo de vivienda:', sorted(df['Tipo_vivienda'].unique()))

        filtro_Distrito = st.sidebar.selectbox('Filtrar por distrito/Distrito:',  sorted(df['Distrito'].unique()))

        filtro_Precio = st.sidebar.slider('Filtrar por precio:', min_value=min_precio, max_value=max_precio, value=(min_precio, max_precio))

        if st.sidebar.button('Aplicar'):
            # st.write('Hola mundo!')
            data_filtrada = df[['Referencia', 'Tipo_vivienda', 'Distrito', 'Precio']][
            (df['Precio'] >= filtro_Precio[0]) & (df['Precio'] <= filtro_Precio[1]) &
            (df['Tipo_vivienda'] == filtro_Tipo_vivienda) & (df['Distrito'] == filtro_Distrito)
            ]
            data_filtrada.reset_index(inplace=True, drop=True)
            st.session_state.current_image_index = 0

        try:
            st.table(data_filtrada.head())
        except:
            st.table(df[['Referencia', 'Tipo_vivienda', 'Distrito', 'Precio']].head())
    return main()

def alquilapisoML():
    st.markdown(
        """
        <h1 style='text-align: center; color: orange;'>Estima el precio de alquiler de tu hogar:</h1>
        """,
        unsafe_allow_html=True
    )

    df_distritos = pd.read_csv(ruta_medidas_alquiler, sep=';')
    df_vivienda = pd.read_csv(ruta_tipovivienda_alquiler, sep=';')
    df_rangos = pd.read_csv(ruta_rangos_alquiler, sep=';')
    # Load the trained model

    import joblib

    model = joblib.load(ruta_modelo_alquiler)
    listazonas = df_distritos['Distrito'].unique()
    listatipovivienda = df_vivienda['TipoVivienda'].unique()
    zona = st.selectbox('Zona', listazonas, index=56)
    tipo_vivienda = st.selectbox('Tipo de vivienda', listatipovivienda, index=0)

    # Definir los widgets de entrada para cada característica
    superficie = st.number_input('Superficie', value=93, min_value=int(df_rangos['Superficie construida'].min()), max_value=int(df_rangos['Superficie construida'].max()), step=1)
    habitaciones = st.selectbox('Habitaciones', list(range(df_rangos['Habitaciones'].min(), df_rangos['Habitaciones'].max())), index=2)
    baños = st.selectbox('Baños', list(range(df_rangos['Baños'].min(), df_rangos['Baños'].max())), index=1)


    conservacion = st.selectbox('Conservación', ['A reformar', 'En buen estado', 'Reformado', 'A estrenar'], index=2)
    lista_plantas = ['Semisótano', 'Bajo', 'Entresuelo', 'Principal', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13']
    planta = st.selectbox('Planta', lista_plantas, index=1)

    def crear_input():

        pmzona = df_distritos[df_distritos['Distrito'] == zona]['PMZona'].values[0]
        pm2zona = df_distritos[df_distritos['Distrito'] == zona]['PMm²Zona'].values[0]
        rmh = df_distritos[df_distritos['Distrito'] == zona]['RMH'].values[0]

        plantaenc = planta.replace('Sótano', '-1').replace('Semisótano', '-0.5').replace('Bajo', '0').replace('Entresuelo', '0').replace('Principal', '0.5')

        conservaciond = conservacion.replace('A reformar', '0').replace('En buen estado', '1').replace('Reformado', '2').replace('A estrenar', '3')

        # Crear DataFrame con los valores ingresados
        X_prueba = pd.DataFrame({
            'Superficie construida': [superficie],
            'Habitaciones': [habitaciones],
            'Baños': [baños],
            'Planta': [plantaenc],
            #'Conservación': [conservacion],
            'PMZona': [pmzona],
            'PMm²Zona': [pm2zona],
            'RMH': [rmh],
            'Conservación_A estrenar': [1 if conservaciond == 'A estrenar' else 0],
            'Conservación_A reformar': [1 if conservaciond == 'A reformar' else 0],
            'Conservación_En buen estado': [1 if conservaciond == 'En buen estado' else 0],
            'Conservación_Reformado': [1 if conservaciond == 'Reformado' else 0],
            'Tipo_vivienda_Apartamento': [1 if tipo_vivienda == 'Apartamento' else 0],
            'Tipo_vivienda_Casa': [1 if tipo_vivienda == 'Casa' else 0],
            'Tipo_vivienda_Casa adosada': [1 if tipo_vivienda == 'Casa adosada' else 0],
            'Tipo_vivienda_Casa pareada': [1 if tipo_vivienda == 'Casa pareada' else 0],
            'Tipo_vivienda_Casa unifamiliar': [1 if tipo_vivienda == 'Casa unifamiliar' else 0],
            'Tipo_vivienda_Chalet': [1 if tipo_vivienda == 'Chalet' else 0],
            'Tipo_vivienda_Chalet adosado': [1 if tipo_vivienda == 'Chalet adosado' else 0],
            'Tipo_vivienda_Chalet pareado': [1 if tipo_vivienda == 'Chalet pareado' else 0],
            'Tipo_vivienda_Chalet unifamiliar': [1 if tipo_vivienda == 'Chalet unifamiliar' else 0],
            'Tipo_vivienda_Dúplex': [1 if tipo_vivienda == 'Dúplex' else 0],
            'Tipo_vivienda_Estudio': [1 if tipo_vivienda == 'Estudio' else 0],
            'Tipo_vivienda_Loft': [1 if tipo_vivienda == 'Loft' else 0],
            'Tipo_vivienda_Piso': [1 if tipo_vivienda == 'Piso' else 0],
            'Tipo_vivienda_Ático': [1 if tipo_vivienda == 'Ático' else 0]
        })
        return model.predict(X_prueba)
        
    # Button to apply the function
    if st.button('Calcular precio'):

        
        predecimos = crear_input()
        precio = format(round(float(predecimos), 2), ',')

        mediapisos = df_rangos['Precio'].mean()

        if predecimos < mediapisos:
            st.balloons()
        else:
            st.snow()


        st.markdown(
            f"""
            <style>
                .blue-text {{
                    color: #0056b3; /* Color azul más oscuro */
                    font-size: 24px; /* Tamaño de fuente más grande */
                    font-family: Arial, sans-serif; /* Tipo de fuente */
                    text-align: center;
                    padding: 20px; /* Añadir espacio alrededor del texto */
                    border-radius: 10px; /* Bordes redondeados */
                    background-color: #e6f2ff; /* Fondo azul claro */
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); /* Sombra suave */
                    width: 60%; /* Ancho del contenedor */
                    margin: auto; /* Centrar horizontalmente */
                }}
            </style>
            <div class="blue-text">Estimamos el precio de alquiler de tu vivienda en: {precio} €</div>
            """,
            unsafe_allow_html=True
        )

        col1, col2, col3 = st.columns(3)
        col2.metric("Precio", f'{precio} €', format(round(float(predecimos)-mediapisos, 2), ',')+' € de diferencia con la media de pisos analizados (ML)', delta_color='inverse')
    
    return crear_input()

def vendepisoML():
    import streamlit as st
    st.markdown(
        """
        <h1 style='text-align: center; color: blue;'>Estima el precio de venta de tu hogar:</h1>
        """,
        unsafe_allow_html=True
    )

    df_distritos = pd.read_csv(ruta_medidas_venta, sep=';')
    df_vivienda = pd.read_csv(ruta_tipovivienda_venta, sep=';')
    df_rangos = pd.read_csv(ruta_rangos_venta, sep=';')

    # Load the trained model
    import joblib
    model = joblib.load(ruta_modelo_venta)
    listazonas = df_distritos['Distrito'].unique()
    listatipovivienda = df_vivienda['TipoVivienda'].unique()





    zona = st.selectbox('Zona', listazonas, index=81)
    tipo_vivienda = st.selectbox('Tipo de vivienda', listatipovivienda, index=0)

    # Definir los widgets de entrada para cada característica
    superficie = st.number_input('Superficie', value=93, min_value=int(df_rangos['Superficie construida'].min()), max_value=int(df_rangos['Superficie construida'].max()), step=1)
    habitaciones = st.selectbox('Habitaciones', list(range(df_rangos['Habitaciones'].min(), df_rangos['Habitaciones'].max())), index=3)
    baños = st.selectbox('Baños', list(range(df_rangos['Baños'].min(), df_rangos['Baños'].max())), index=2)



    conservacion = st.selectbox('Conservación', ['A reformar', 'En buen estado', 'Reformado', 'A estrenar'], index=2)


    lista_plantas = ['Sótano', 'Semisótano', 'Bajo', 'Entresuelo', 'Principal', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13']

    planta = st.selectbox('Planta', lista_plantas, index=2)


    def crear_input():

        pmzona = df_distritos[df_distritos['Distrito'] == zona]['PMZona'].values[0]
        pm2zona = df_distritos[df_distritos['Distrito'] == zona]['PMm²Zona'].values[0]
        rmh = df_distritos[df_distritos['Distrito'] == zona]['RMH'].values[0]

        plantaenc = planta.replace('Sótano', '-1').replace('Semisótano', '-0.5').replace('Bajo', '0').replace('Entresuelo', '0').replace('Principal', '0.5')

        conservaciond = conservacion.replace('A reformar', '0').replace('En buen estado', '1').replace('Reformado', '2').replace('A estrenar', '3')


        # Crear DataFrame con los valores ingresados
        X_prueba = pd.DataFrame({
            'Superficie construida': [superficie],
            'Habitaciones': [habitaciones],
            'Baños': [baños],
            'Planta': [plantaenc],
            'Conservación': [conservaciond],
            'PMZona': [pmzona],
            'PMm²Zona': [pm2zona],
            'RMH': [rmh],
            'Tipo_vivienda_Apartamento': [1 if tipo_vivienda == 'Apartamento' else 0],
            'Tipo_vivienda_Casa': [1 if tipo_vivienda == 'Casa' else 0],
            'Tipo_vivienda_Casa adosada': [1 if tipo_vivienda == 'Casa adosada' else 0],
            'Tipo_vivienda_Casa pareada': [1 if tipo_vivienda == 'Casa pareada' else 0],
            #'Tipo_vivienda_Casa rústica': [1 if tipo_vivienda == 'Casa rústica' else 0],
            'Tipo_vivienda_Casa unifamiliar': [1 if tipo_vivienda == 'Casa unifamiliar' else 0],
            'Tipo_vivienda_Chalet': [1 if tipo_vivienda == 'Chalet' else 0],
            'Tipo_vivienda_Chalet adosado': [1 if tipo_vivienda == 'Chalet adosado' else 0],
            'Tipo_vivienda_Chalet pareado': [1 if tipo_vivienda == 'Chalet pareado' else 0],
            #'Tipo_vivienda_Chalet unifamiliar': [1 if tipo_vivienda == 'Chalet unifamiliar' else 0],
            'Tipo_vivienda_Dúplex': [1 if tipo_vivienda == 'Dúplex' else 0],
            #'Tipo_vivienda_Estudio': [1 if tipo_vivienda == 'Estudio' else 0],
            'Tipo_vivienda_Loft': [1 if tipo_vivienda == 'Loft' else 0],
            'Tipo_vivienda_Piso': [1 if tipo_vivienda == 'Piso' else 0],
            'Tipo_vivienda_Ático': [1 if tipo_vivienda == 'Ático' else 0]
        })


        return model.predict(X_prueba)


    # Button to apply the function
    if st.button('Calcular precio'):

        predecimos = crear_input()
        precio = format(round(float(predecimos), 2), ',')

        mediapisos = df_rangos['Precio'].mean()

        if predecimos < mediapisos:
            st.balloons()
        else:
            st.snow()
        
        st.markdown(
            f"""
            <style>
                .blue-text {{
                    color: #0056b3; /* Color azul más oscuro */
                    font-size: 24px; /* Tamaño de fuente más grande */
                    font-family: Arial, sans-serif; /* Tipo de fuente */
                    text-align: center;
                    padding: 20px; /* Añadir espacio alrededor del texto */
                    border-radius: 10px; /* Bordes redondeados */
                    background-color: #e6f2ff; /* Fondo azul claro */
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); /* Sombra suave */
                    width: 60%; /* Ancho del contenedor */
                    margin: auto; /* Centrar horizontalmente */
                }}
            </style>
            <div class="blue-text">Estimamos el precio de venta de tu vivienda en: {precio} €</div>
            """,
            unsafe_allow_html=True
        )
        
        col1, col2, col3 = st.columns(3)
        col2.metric("Precio", f'{precio} €', format(round(float(predecimos)-mediapisos, 2), ',')+' € de diferencia con la media de pisos analizados (ML)', delta_color='inverse')

    return crear_input()


#Funcion Mapa
def abrir_mapa():
    df_alquiler = pd.read_csv(ruta_alquileres_coordenadas, sep=';', header=0)
    df_venta = pd.read_csv(ruta_ventas_coordenadas, sep = ';', header=0)

    df_distritos = gpd.read_file(ruta_distritos_geo)
    df_municipios = gpd.read_file(ruta_municipios_geo)

    df_paradas =gpd.read_file(ruta_paradas_metro_geo)
    df_lineas = gpd.read_file(ruta_lineas_metro_geo)
    df_sanidad=pd.read_csv(ruta_sanidad, sep='|', header=0)
    df_bibliotecas = pd.read_csv(ruta_bibliotecas, sep=';' , header=0)
    df_polideportivos = pd.read_csv(ruta_polideportivos, sep=';' , header=0)
    df_colegios = pd.read_csv(ruta_colegios, sep=';', header=0)
    df_museos = pd.read_csv(ruta_museos, sep=';', header=0)
    df_universidades = pd.read_csv(ruta_universidades, sep=';', header=0)

    config = {
        'version': 'v1',
        'config': {
            'mapState':{
                'latitude': 40.4167278,
                'longitude': -3.7033387,
                'zoom': 15 
                }
            }
    }

    with open(ruta_config_leer, 'r') as j:
        configuracion = json.load(j)
    type(configuracion)



    mapa = KeplerGl(show_docs=False, config=configuracion)
    mapa.add_data(data=df_alquiler, name='VIVIENDAS EN ALQUILER')
    mapa.add_data(data=df_venta, name='VIVIENDAS EN VENTA')
    mapa.add_data(data=df_paradas, name='PARADAS DE METRO')

    mapa.add_data(data=df_distritos, name='DISTRITOS DE MADRID')
    mapa.add_data(data=df_municipios, name='MUNICIPIOS')

    mapa.add_data(data=df_lineas, name='LINEAS DE METRO')
    mapa.add_data(data=df_sanidad, name='ATENCION MEDICA')
    mapa.add_data(data=df_bibliotecas, name='BIBLIOTECAS')
    mapa.add_data(data=df_polideportivos, name='POLIDEPORTIVOS')
    mapa.add_data(data=df_colegios, name='COLEGIOS')
    mapa.add_data(data=df_museos, name='MUSEOS')
    mapa.add_data(data=df_universidades, name='CENTROS DE ENSENANZA SUPERIOR')



    st.markdown("<h1 style='text-align: center; color: black;'>Mapa de Madrid con los datos de viviendas, transporte, educación, sanidad, cultura y deporte</div>", unsafe_allow_html=True)
    keplergl_static(mapa, height=600, width=1400)

    #if st.button('Guardar configuración'):
    #    with open(ruta_config_guardar, 'w') as file:
    #        json.dump(mapa.config, file, indent=4, sort_keys=True)

    #    st.warning('Configuración guardada')


    return config

def chatbot():
    with st.sidebar:
        st.title("🦙💬 Chat bot")
        st.header("Settings")

        add_replicate_api=st.text_input('Enter the Replicate API token', type='password')
        if not (add_replicate_api.startswith('r8_') and len(add_replicate_api)==40):
            st.warning('Please enter your credentials', icon='⚠️')
        else:
            st.success('Proceed to entering your prompt message!', icon='👉')

        st.subheader("Models and Parameters")

        select_model=st.selectbox("Choose a Llama 2 Model", ['Llama 2 7b', 'Llama 2 13b','llama prueba','llama prueba2', 'Llama 2 70b'], key='select_model')
        if select_model=='Llama 2 7b':
            llm = 'a16z-infra/llama7b-v2-chat:4f0a4744c7295c024a1de15e1a63c880d3da035fa1f49bfd344fe076074c8eea'
        elif select_model=='Llama 2 13b':
            llm = 'meta/llama-2-13b-chat:f4e2de70d66816a838a89eeeb621910adffb0dd0baba3976c96980970978018d'
        elif select_model=='Llama 2 prueba':
            llm = 'replicate/llama-7b:455d66312a66299fba685548fe24f66880f093007b927abd19f4356295f8577c'
        elif select_model=='Llama 2 prueba2 esp':
            llm = 'replicate/llama-7b:ac808388e2e9d8ed35a5bf2eaa7d83f0ad53f9e3df31a42e4eb0a0c3249b3165'        
        else:
            llm = 'replicate/llama70b-v2-chat:e951f18578850b652510200860fc4ea62b3b16fac280f83ff32282f87bbd2e48'

        temperature=st.slider('temperature', min_value=0.01, max_value=5.0, value=0.1, step=0.01)
        top_p=st.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
        max_length=st.slider('max_length', min_value=64, max_value=4096, value=512, step=8)

    os.environ['REPLICATE_API_TOKEN']=add_replicate_api

    #Store the LLM Generated Reponese

    if "messages" not in st.session_state.keys():
        st.session_state.messages=[{"role": "assistant", "content":"How may I assist you today?"}]

    # Diplay the chat messages

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])


    # Clear the Chat Messages
    def clear_chat_history():
        st.session_state.messages=[{"role":"assistant", "content": "How may I assist you today"}]

    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    # Create a Function to generate the Llama 2 Response
    def generate_llama2_response(prompt_input):
        default_system_prompt="You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
        for data in st.session_state.messages:
            print("Data:", data)
            if data["role"]=="user":
                default_system_prompt+="User: " + data["content"] + "\n\n"
            else:
                default_system_prompt+="Assistant" + data["content"] + "\n\n"
        output=replicate.run(llm, input={"prompt": f"{default_system_prompt} {prompt_input} Assistant: ",
                                        "temperature": temperature, "top_p":top_p, "max_length": max_length, "repititon_penalty":1})

        return output


    #User -Provided Prompt

    if prompt := st.chat_input(disabled=not add_replicate_api):
        st.session_state.messages.append({"role": "user", "content":prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Generate a New Response if the last message is not from the asssistant

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response=generate_llama2_response(prompt)
                placeholder=st.empty()
                full_response=''
                for item in response:
                    full_response+=item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)

        message= {"role":"assistant", "content":full_response}
        st.session_state.messages.append(message)




#Pestañas de pagina
if selected =="Inicio":
    HomePage()


if selected == "Inmuebles":
    df_selection = Inmuebles()
    try:
        st.markdown("""---""") #Diferencia entre dos
        total_tipo_vivienda = float(df_selection['Tipo_vivienda'].count())
        vivienda_mode = df_selection['Tipo_vivienda'].mode().iloc[0]

        total1,total2 = st.columns(2,gap='large')
        with total1:
            st.info('Total de habitantes en Madrid 👫', icon="🔍")
            st.metric(label = 'nº habitantes', value= f"{total_tipo_vivienda:,.0f}")

        with total2:
            st.info('Tipo de vivienda que mas se encuentra 🏘', icon="🔍")
            st.metric(label='Tipo vivienda', value=f"{vivienda_mode}")
    except:
            st.warning("Empiece escogiendo entre Alquiler o Venta ")


if selected == "Ofertas":
    ofertas()

if selected == "Alquiler":
    st.title(f"You have selected {selected}")

if selected=="Alquila tu piso":
    alquilapisoML()

if selected=="Vende tu piso":
    vendepisoML()


if selected == "Mapa":
    abrir_mapa()

if selected== "Chatbot":
    chatbot()

footer = """
<style>
a:hover,  a:active {
    color: red;
    background-color: transparent;
    text-decoration: underline;
}

.footer {
    position: fixed;
    left: 0;
    height: 12%;
    bottom: 0;
    width: 100%;
    background-color: #243946;
    color: black;
    text-align: center;
}

.footer img {
    width: 30px; /* ajusta el tamaño según sea necesario */
    height: auto;
    margin: 20px; /* ajusta el margen según sea necesario */
}

.footer p {
    margin: 10px; /* ajusta el margen según sea necesario */
}

.social-icons {
    display: inline-block;
}
</style>
<div class="footer">
    <div class="social-icons">
        <p>
            <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/95/Instagram_logo_2022.svg/1200px-Instagram_logo_2022.svg.png" alt="Instagram" title="Instagram">
            Síguenos en <a href="https://www.instagram.com/" target="_blank">Instagram</a>
        </p>
    </div>
    <div class="social-icons">
        <p>
            <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b8/2021_Facebook_icon.svg/220px-2021_Facebook_icon.svg.png" alt="Facebook" title="Facebook">
            Síguenos en <a href="https://www.facebook.com/" target="_blank">Facebook</a>
        </p>
    </div>
    <div class="social-icons">
        <p>
            <img src="https://store-images.s-microsoft.com/image/apps.31120.9007199266245564.44dc7699-748d-4c34-ba5e-d04eb48f7960.bc4172bd-63f0-455a-9acd-5457f44e4473" alt="LinkedIn" title="LinkedIn">
            Síguenos en <a href="https://www.linkedin.com/" target="_blank">LinkedIn</a>
        </p>
    </div>
</div>
"""

st.markdown(footer, unsafe_allow_html=True)