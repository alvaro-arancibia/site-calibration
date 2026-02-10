import streamlit as st
import pandas as pd
import io
import numpy as np

# Core Imports for Offline Processing
from sitecal.core.calibration_engine import Similarity2D
from sitecal.core.projections import ProjectionFactory
from sitecal.infrastructure.reports import generate_markdown_report

def validate_collinearity(df: pd.DataFrame) -> bool:
    """Checks for collinearity in points."""
    if "Easting_global" not in df.columns or "Northing_global" not in df.columns:
        return False
    coords = df[["Easting_global", "Northing_global"]].values
    if len(coords) < 3: return False
    centered = coords - np.mean(coords, axis=0)
    cov = np.cov(centered, rowvar=False)
    eigvals = np.linalg.eigvals(cov)
    if np.max(eigvals) == 0: return True
    return (np.min(eigvals) / np.max(eigvals)) < 1e-4

def main():
    st.set_page_config(page_title="Site Calibration (Offline)", page_icon="🛰️", layout="wide")
    
    st.title("Site Calibration Tool (Monolith)")
    st.markdown("Cálculo local y seguro. No requiere conexión a internet.")

    # Instructions
    with st.expander("ℹ️ Instrucciones de Formato CSV (Importante)"):
        st.markdown("""
        ### Archivo Global (GNSS)
        **Formato:** Coordenadas Geodésicas WGS84 (Grados Decimales)
        * **Columnas Requeridas:** `Point` (ID), `Latitude`, `Longitude`, `Ellipsoidal Height` (o `h`)
        * **Precisión:** Al menos **8 decimales** en Lat/Lon para asegurar precisión milimétrica.

        ### Archivo Local (Planas)
        **Formato:** Coordenadas Cartesianas Locales (Metros)
        * **Columnas Requeridas:** `Point` (ID), `Easting` (Este), `Northing` (Norte), `Elevation` (o `z`, `h`)
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
            
            # Helper Visualization
            st.markdown("##### Geometría Local")
            try:
                # Simple scatter of local coords
                chart_data = local_df.rename(columns={l_e: "Easting", l_n: "Northing"})
                st.scatter_chart(chart_data, x="Easting", y="Northing", color="#FF4B4B")
            except Exception:
                st.caption("No se pudo generar la previsualización gráfica.")

    # Method Selection and Parameters
    st.subheader("Método y Parámetros")
    col_method, col_params = st.columns([1, 3])
    
    with col_method:
        # Only supporting Similarity2D for now as per instructions (Default/LTM map to it internally anyway)
        # But User asked for Similarity2D specifically.
        # Keeping selection for UI consistency if they want to label it, but logic will force Similarity2D
        method = st.selectbox("Seleccionar Método", ["Default", "LTM"])
    
    params = {}
    if method == "LTM":
        with col_params:
            c1, c2, c3, c4 = st.columns(4)
            with c1: params["central_meridian"] = st.number_input("Meridiano Central", value=-72.0)
            with c2: params["scale_factor"] = st.number_input("Factor de Escala", value=0.9996, format="%.6f")
            with c3: params["false_easting"] = st.number_input("Falso Este", value=500000.0)
            with c4: params["false_northing"] = st.number_input("Falso Norte", value=10000000.0)

    # Action
    st.markdown("---")
    if st.button("Calcular Calibración (Offline)", type="primary", use_container_width=True):
        if global_df is None or local_df is None:
            st.error("Por favor sube ambos archivos CSV (Global y Local).")
            return

        with st.spinner("Procesando localmente..."):
            try:
                # 1. Standardize Inputs (Strict naming for Core)
                df_g_ready = global_df.rename(columns={
                    g_point: "Point", g_lat: "Latitude", g_lon: "Longitude", g_h: "EllipsoidalHeight"
                })[["Point", "Latitude", "Longitude", "EllipsoidalHeight"]]
                # Ensure Point is string
                df_g_ready["Point"] = df_g_ready["Point"].astype(str)

                df_l_ready = local_df.rename(columns={
                    l_point: "Point", l_e: "Easting", l_n: "Northing", l_z: "Elevation"
                })[["Point", "Easting", "Northing", "Elevation"]]
                df_l_ready["Point"] = df_l_ready["Point"].astype(str)

                # 2. Projection
                proj_params = {k: v for k, v in params.items()}
                projection = ProjectionFactory.create(method.lower(), **proj_params)
                df_g_proj = projection.project(df_g_ready)

                # 3. Merge
                merged_df = pd.merge(df_l_ready, df_g_proj, on="Point", suffixes=('_local', '_global'))
                if len(merged_df) < 3:
                    st.error(f"Error: Solo se encontraron {len(merged_df)} puntos comunes. Se requieren mínimo 3.")
                    return
                
                if validate_collinearity(merged_df):
                    st.error("Error: Los puntos son colineales o geográficamente muy cercanos. Geometría inestable.")
                    return

                # 4. Calibration Engine
                # DIRECT INSTANTIATION AS REQUESTED
                engine = Similarity2D()
                engine.train(df_l_ready, df_g_proj)

                # 5. Build Result Object (Mimicking API response structure for reuse)
                residuals = [
                    {"Point": str(row["Point"]), "dE": float(row["dE"]), "dN": float(row["dN"]), "dH": float(row["dH"])}
                    for _, row in engine.residuals.iterrows()
                ]
                
                report_text = generate_markdown_report(engine, "not_used", method.lower())

                result_data = {
                    "parameters": {
                        "horizontal": engine.horizontal_params,
                        "vertical": engine.vertical_params
                    },
                    "residuals": residuals,
                    "report": report_text
                }

                display_results(result_data)

                # Save calibration state for transforming new points
                st.session_state["calibration_engine"] = engine
                st.session_state["calibration_projection"] = projection

            except Exception as e:
                st.error(f"Error Interno: {str(e)}")

    # --- Transform New Points Section ---
    if "calibration_engine" in st.session_state and "calibration_projection" in st.session_state:
        st.markdown("---")
        st.subheader("Transformar Puntos Nuevos")
        st.markdown("Sube un CSV con puntos GPS nuevos para obtener sus coordenadas locales usando la calibracion activa.")

        new_file = st.file_uploader("Subir CSV de Puntos Nuevos", type=["csv"], key="new_points")
        if new_file:
            has_header_n = st.checkbox("Tiene encabezados", value=True, key="header_n")
            new_df = pd.read_csv(new_file, header=0 if has_header_n else None)
            st.dataframe(new_df.head(), use_container_width=True)

            st.markdown("##### Mapeo de Columnas")
            cols_n = new_df.columns.tolist()
            col_a, col_b, col_c, col_d = st.columns(4)
            with col_a:
                n_point = st.selectbox("Point (ID)", cols_n, index=0 if cols_n else 0, key="n_pt")
            with col_b:
                n_lat = st.selectbox("Latitude", cols_n, index=1 if len(cols_n) > 1 else 0, key="n_lat")
            with col_c:
                n_lon = st.selectbox("Longitude", cols_n, index=2 if len(cols_n) > 2 else 0, key="n_lon")
            with col_d:
                n_h = st.selectbox("Ellipsoidal Height", cols_n, index=3 if len(cols_n) > 3 else 0, key="n_h")

            if st.button("Transformar", type="primary"):
                try:
                    df_n_ready = new_df.rename(columns={
                        n_point: "Point", n_lat: "Latitude", n_lon: "Longitude", n_h: "EllipsoidalHeight"
                    })[["Point", "Latitude", "Longitude", "EllipsoidalHeight"]]
                    df_n_ready["Point"] = df_n_ready["Point"].astype(str)

                    proj = st.session_state["calibration_projection"]
                    eng = st.session_state["calibration_engine"]

                    df_n_proj = proj.project(df_n_ready)
                    transformed = eng.transform(df_n_proj)

                    st.subheader("Resultados de Transformacion")
                    st.dataframe(transformed.rename(columns={"h": "Elevation"}), use_container_width=True)

                    csv_out = transformed.rename(columns={"h": "Elevation"}).to_csv(index=False)
                    st.download_button(
                        label="Descargar CSV",
                        data=csv_out,
                        file_name="transformed_points.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    st.error(f"Error al transformar: {str(e)}")

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
