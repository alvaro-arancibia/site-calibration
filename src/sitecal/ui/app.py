import streamlit as st
import pandas as pd
import requests
import io

def main():
    st.set_page_config(page_title="Calibración de Obra", page_icon="🛰️", layout="wide")
    
    st.title("Herramienta de Calibración de Obra")
    st.markdown("Sube tus coordenadas globales (GNSS) y locales (Planas) para calcular los parámetros de calibración.")

    # Sidebar for API Configuration
    st.sidebar.header("Configuración")
    api_url = st.sidebar.text_input("URL de la API", value="https://site-calibration-api-845253769460.us-central1.run.app/calibrate")
    
    # Instructions
    with st.expander("ℹ️ Instrucciones de Formato CSV (Importante)"):
        st.markdown("""
        ### Archivo Global (GNSS)
        **Formato:** Coordenadas Geodésicas WGS84 (Grados Decimales)
        * **Columnas Requeridas:** `Point` (ID), `Latitude`, `Longitude`, `Ellipsoidal Height` (o `h`)
        * **Precisión:** Al menos **8 decimales** en Lat/Lon para asegurar precisión milimétrica.
        * **Ejemplo:**
        ```csv
        Point,Latitude,Longitude,h
        100,-33.45678912, -70.65432198, 500.123
        ```

        ### Archivo Local (Planas)
        **Formato:** Coordenadas Cartesianas Locales (Metros)
        * **Columnas Requeridas:** `Point` (ID), `Easting` (Este), `Northing` (Norte), `Elevation` (o `z`, `h`)
        * **Ejemplo:**
        ```csv
        Point,Easting,Northing,Elevation
        100,500123.456,1000789.012,100.456
        ```
        *Nota: El ID (Point) debe coincidir exactamente en ambos archivos.*
        """)
        
    # Main Input Section
    col1, col2 = st.columns(2)
    
    global_df = None
    local_df = None
    
    with col1:
        st.subheader("Coordenadas Globales (CSV)")
        global_file = st.file_uploader("Subir CSV Global", type=["csv"], key="global")
        if global_file:
            has_header_g = st.checkbox("Tiene encabezados", value=True, key="header_g")
            global_df = pd.read_csv(global_file, header=0 if has_header_g else None)
            st.dataframe(global_df.head(), use_container_width=True)
            
            st.markdown("##### Mapeo de Columnas")
            cols_g = global_df.columns.tolist()
            g_point = st.selectbox("Point (ID)", cols_g, index=0 if cols_g else 0, key="g_pt")
            g_lat = st.selectbox("Latitude", cols_g, index=1 if len(cols_g)>1 else 0, key="g_lat")
            g_lon = st.selectbox("Longitude", cols_g, index=2 if len(cols_g)>2 else 0, key="g_lon")
            g_h = st.selectbox("Ellipsoidal Height", cols_g, index=3 if len(cols_g)>3 else 0, key="g_h")
        
    with col2:
        st.subheader("Coordenadas Locales (CSV)")
        local_file = st.file_uploader("Subir CSV Local", type=["csv"], key="local")
        if local_file:
            has_header_l = st.checkbox("Tiene encabezados", value=True, key="header_l")
            local_df = pd.read_csv(local_file, header=0 if has_header_l else None)
            st.dataframe(local_df.head(), use_container_width=True)

            st.markdown("##### Mapeo de Columnas")
            cols_l = local_df.columns.tolist()
            l_point = st.selectbox("Point (ID)", cols_l, index=0 if cols_l else 0, key="l_pt")
            l_e = st.selectbox("Easting", cols_l, index=1 if len(cols_l)>1 else 0, key="l_e")
            l_n = st.selectbox("Northing", cols_l, index=2 if len(cols_l)>2 else 0, key="l_n")
            l_z = st.selectbox("Elevation", cols_l, index=3 if len(cols_l)>3 else 0, key="l_z")


    # Method Selection and Parameters
    st.subheader("Método y Parámetros")
    col_method, col_params = st.columns([1, 3])
    
    with col_method:
        method = st.selectbox("Seleccionar Método", ["TBC", "LTM"])
    
    params = {}
    if method == "LTM":
        with col_params:
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                params["central_meridian"] = st.number_input("Meridiano Central", value=-72.0)
            with c2:
                params["scale_factor"] = st.number_input("Factor de Escala", value=0.9996, format="%.6f")
            with c3:
                params["false_easting"] = st.number_input("Falso Este", value=500000.0)
            with c4:
                params["false_northing"] = st.number_input("Falso Norte", value=10000000.0)

    # Action
    st.markdown("---")
    if st.button("Calcular Calibración", type="primary", use_container_width=True):
        if global_df is None or local_df is None:
            st.error("Por favor sube ambos archivos CSV (Global y Local).")
            return

        with st.spinner("Calculando parámetros de calibración..."):
            try:
                # Prepare Global CSV
                df_g_ready = global_df.rename(columns={
                    g_point: "Point",
                    g_lat: "Latitude",
                    g_lon: "Longitude", 
                    g_h: "EllipsoidalHeight"
                })[["Point", "Latitude", "Longitude", "EllipsoidalHeight"]]
                
                # Prepare Local CSV
                df_l_ready = local_df.rename(columns={
                    l_point: "Point",
                    l_e: "Easting",
                    l_n: "Northing",
                    l_z: "Elevation"
                })[["Point", "Easting", "Northing", "Elevation"]]

                # Convert to CSV in memory
                buffer_g = io.StringIO()
                df_g_ready.to_csv(buffer_g, index=False)
                
                buffer_l = io.StringIO()
                df_l_ready.to_csv(buffer_l, index=False)

                files = {
                    "global_csv": ("global.csv", buffer_g.getvalue(), "text/csv"),
                    "local_csv": ("local.csv", buffer_l.getvalue(), "text/csv"),
                }
                
                # Prepare query parameters
                query_params = {"method": method.lower()} # Ensure lower case for API
                if method == "LTM":
                    query_params.update(params)

                # Make API request
                response = requests.post(api_url, files=files, params=query_params)
                
                if response.status_code == 200:
                    data = response.json()
                    display_results(data)
                else:
                    st.error(f"Error de API ({response.status_code}): {response.text}")
            except requests.exceptions.ConnectionError:
                 st.error("No se pudo conectar a la API. Por favor verifica la URL.")
            except Exception as e:
                st.error(f"Ocurrió un error: {str(e)}")

def display_results(data):
    # 1. Calculated Parameters
    if "parameters" in data:
        p = data["parameters"]
        
        # Horizontal
        if "horizontal" in p and p["horizontal"]:
             st.subheader("🏗️ Ajuste Horizontal (2D)")
             hp = p["horizontal"]
             c1, c2, c3, c4 = st.columns(4)
             with c1: st.metric("Factor Escala (a)", f"{hp['a']:.7f}")
             with c2: st.metric("Rotación (b)", f"{hp['b']:.7f}")
             with c3: st.metric("Traslasión Este", f"{hp['tE']:.3f} m")
             with c4: st.metric("Traslasión Norte", f"{hp['tN']:.3f} m")
             
        # Vertical
        if "vertical" in p and p["vertical"]:
             st.subheader("📐 Ajuste Vertical (1D)")
             vp = p["vertical"]
             c1, c2, c3, c4 = st.columns(4)
             with c1: st.metric("Shift Vertical", f"{vp['vertical_shift']:.3f} m")
             with c2: st.metric("Inclinación N", f"{vp['slope_north']*1e6:.2f} ppm")
             with c3: st.metric("Inclinación E", f"{vp['slope_east']*1e6:.2f} ppm")
             with c4: st.metric("Centroide", f"({vp['centroid_north']:.0f}, {vp['centroid_east']:.0f})")

        st.markdown("---")

    # 2. Residuals Table
    if "residuals" in data:
        st.subheader("Cuadrícula de Residuales")
        residuals = data["residuals"]
        if isinstance(residuals, list) and len(residuals) > 0:
             df = pd.DataFrame(residuals)
             # Rename columns for display
             df.rename(columns={"dE": "dE (m)", "dN": "dN (m)", "dH": "dH (m)"}, inplace=True)
             st.dataframe(df, use_container_width=True)
        else:
            st.info("No se devolvieron datos de residuales.")
        st.markdown("---")

    # 3. Full Report
    if "report" in data:
        st.subheader("Reporte Completo de Calibración")
        st.markdown(data["report"])
    elif "markdown_report" in data: 
        st.subheader("Reporte Completo de Calibración")
        st.markdown(data["markdown_report"])

if __name__ == "__main__":
    main()
