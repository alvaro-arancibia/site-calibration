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
            
            use_wls = st.checkbox("Ingresar precisión por punto (WLS)", value=False, key="use_wls")
            if use_wls:
                if "sigma" not in local_df.columns:
                    local_df["sigma"] = 1.0  # Default to 1.0 m precision
                st.caption("Edita la columna 'sigma' para aplicar pesos personalizados (WLS):")
                local_df = st.data_editor(local_df, num_rows="dynamic", key="local_editor")
            else:
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

                # Mapear las columnas que existen
                cols_to_keep_l = ["Point", "Easting", "Northing", "Elevation"]
                if "sigma" in local_df.columns and use_wls:
                    cols_to_keep_l.append("sigma")
                    
                df_l_ready = local_df.rename(columns={
                    l_point: "Point", l_e: "Easting", l_n: "Northing", l_z: "Elevation"
                })[cols_to_keep_l]
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
                import warnings
                engine = Similarity2D()
                
                # Capture accurate parameters for serialization
                engine.proj_method = method.lower()
                adj_params = proj_params.copy()
                if engine.proj_method == "default":
                    adj_params["lat_0"] = float(df_g_ready.iloc[0]["Latitude"])
                    adj_params["lon_0"] = float(df_g_ready.iloc[0]["Longitude"])
                elif engine.proj_method == "utm":
                    lon_mean = df_g_ready["Longitude"].mean()
                    utm_zone = int((lon_mean + 180) / 6) + 1
                    is_south = df_g_ready["Latitude"].mean() < 0
                    adj_params["utm_zone"] = utm_zone
                    adj_params["is_south"] = is_south
                elif engine.proj_method == "ltm":
                    adj_params["latitude_of_origin"] = 0.0
                engine.proj_params = adj_params
                
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    engine.train(df_l_ready, df_g_proj)
                    
                    for warn in w:
                        if "Extrapolación detectada" in str(warn.message):
                            st.warning(str(warn.message), icon="⚠️")

                # 5. Build Result Object (Mimicking API response structure for reuse)
                residuals = []
                for _, row in engine.residuals.iterrows():
                    res_dict = {
                        "Point": str(row["Point"]), 
                        "dE": float(row["dE"]), 
                        "dN": float(row["dN"]), 
                        "dH": float(row["dH"])
                    }
                    if "outlier_horizontal" in engine.residuals.columns:
                        res_dict["outlier_horizontal"] = bool(row.get("outlier_horizontal", False))
                    if "outlier_vertical" in engine.residuals.columns:
                        res_dict["outlier_vertical"] = bool(row.get("outlier_vertical", False))
                    residuals.append(res_dict)
                
                report_text = generate_markdown_report(engine, "not_used", method.lower())
                
                result_data = {
                    "parameters": {
                        "horizontal": engine.horizontal_params,
                        "vertical": engine.vertical_params
                    },
                    "residuals": residuals,
                    "report": report_text
                }
                
                st.session_state["cal_engine"] = engine
                st.session_state["cal_result"] = result_data
                
                display_results(result_data)
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.download_button(
                    label="💾 Descargar Calibración (.sitecal)",
                    data=engine.save(),
                    file_name="calibracion.sitecal",
                    mime="application/json",
                    use_container_width=True
                )

            except Exception as e:
                st.error(f"Error Interno: {str(e)}")

    # -------------------------------------------------------------
    # TRANSFORMATION SECTION
    # -------------------------------------------------------------
    st.markdown("---")
    st.header("Transformar Puntos")
    st.markdown("Aplica una calibración existente a un nuevo conjunto de puntos.")
    
    # 1. Provide calibration model source
    load_col, _ = st.columns(2)
    with load_col:
        st.subheader("Modelo de Calibración")
        cal_file = st.file_uploader("Cargar Calibración (.sitecal)", type=["sitecal", "json"])
        
        loaded_engine = None
        if cal_file:
            try:
                content = cal_file.read().decode('utf-8')
                loaded_engine = Similarity2D.load(content)
                st.success("✅ Modelo cargado desde archivo.")
            except Exception as e:
                st.error(f"Error cargando archivo: {str(e)}")
        elif st.session_state.get("cal_engine") is not None:
             loaded_engine = st.session_state["cal_engine"]
             st.info("ℹ️ Usando el modelo de calibración calculado en la sesión actual.")
        else:
             st.warning("⚠️ Debes calcular una calibración arriba o subir un archivo `.sitecal`.")
             
    if loaded_engine is not None:
        # 2. Upload Points
        st.subheader("Puntos a Transformar")
        trans_file = st.file_uploader("Subir CSV de Puntos", type=["csv"], key="trans_points")
        
        if trans_file:
            trans_df = pd.read_csv(trans_file)
            st.dataframe(trans_df.head(), use_container_width=True)
            
            direction = st.radio("Dirección de Transformación", ["Global -> Local", "Local -> Global"])
            
            # Map columns
            tc = trans_df.columns.tolist()
            t_id = st.selectbox("Point (ID)", tc, index=0, key="t_id")
            if direction == "Global -> Local":
                 t_x = st.selectbox("Easting Global / Longitude", tc, index=1 if len(tc)>1 else 0, key="t_g_e")
                 t_y = st.selectbox("Northing Global / Latitude", tc, index=2 if len(tc)>2 else 0, key="t_g_n")
                 t_z = st.selectbox("Ellipsoidal Height / h", tc, index=3 if len(tc)>3 else 0, key="t_g_h")
            else:
                 t_x = st.selectbox("Easting (Local)", tc, index=1 if len(tc)>1 else 0, key="t_l_e")
                 t_y = st.selectbox("Northing (Local)", tc, index=2 if len(tc)>2 else 0, key="t_l_n")
                 t_z = st.selectbox("Elevation (Local)", tc, index=3 if len(tc)>3 else 0, key="t_l_h")
                 
            if st.button("Aplicar Transformación", type="primary", key="btn_trans"):
                 with st.spinner("Transformando..."):
                      try:
                          if direction == "Global -> Local":
                              # Bug 1: project Lat/Lon → metres using the same CRS as training
                              df_ready = trans_df.rename(columns={
                                  t_id: "Point", t_x: "Longitude", t_y: "Latitude", t_z: "EllipsoidalHeight"
                              })[["Point", "Longitude", "Latitude", "EllipsoidalHeight"]]
                              df_ready["Point"] = df_ready["Point"].astype(str)

                              if loaded_engine.proj_method:
                                  from pyproj import CRS, Transformer as _T
                                  _src, _dst = loaded_engine.get_crs_strings()
                                  if _dst:
                                      _tf = _T.from_crs(CRS(_src), CRS(_dst), always_xy=True)
                                      df_ready["Easting_global"], df_ready["Northing_global"] = \
                                          _tf.transform(df_ready["Longitude"].values,
                                                        df_ready["Latitude"].values)
                                  else:
                                      df_ready = df_ready.rename(columns={
                                          "Longitude": "Easting_global", "Latitude": "Northing_global"})
                              else:
                                  df_ready = df_ready.rename(columns={
                                      "Longitude": "Easting_global", "Latitude": "Northing_global"})

                              res_df = loaded_engine.transform(df_ready)
                          else:
                              # Bug 3: subset columns after rename to avoid phantom duplicates
                              df_ready = trans_df.rename(columns={
                                  t_id: "Point", t_x: "Easting_local", t_y: "Northing_local", t_z: "Elevation"
                              })[["Point", "Easting_local", "Northing_local", "Elevation"]]
                              df_ready["Point"] = df_ready["Point"].astype(str)

                              res_df = loaded_engine.transform_inverse(df_ready)
                              
                          st.success("Transformación exitosa.")
                          st.dataframe(res_df.head(), use_container_width=True)
                          
                          csv_data = res_df.to_csv(index=False).encode('utf-8')
                          st.download_button(
                              label="📥 Descargar Resultados CSV",
                              data=csv_data,
                              file_name="puntos_transformados.csv",
                              mime="text/csv"
                          )
                      except Exception as e:
                          st.error(f"Error transformando puntos: {str(e)}")

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

    # 2. Fit Quality Section
    if "residuals" in data and "parameters" in data:
        p = data["parameters"]
        hp = p.get("horizontal", {})
        
        # Determine sigma0 (a posteriori). Fallback to standard error estimation if not exposed from motor
        sigma0_sq = hp.get("sigma0_sq_h")
        residuals_list = data["residuals"]
        
        if residuals_list:
            if sigma0_sq is not None:
                sigma0 = np.sqrt(sigma0_sq)
            else:
                # Approximation of empirical standard deviation (dof = 2n - 4)
                df_res_tmp = pd.DataFrame(residuals_list)
                dof = max(1, 2 * len(df_res_tmp) - 4)
                v_sq = df_res_tmp["dE"]**2 + df_res_tmp["dN"]**2
                sigma0 = np.sqrt(v_sq.sum() / dof)
                
            st.subheader("🎯 Calidad del Ajuste (Horizontal)")
            # Color logic
            if sigma0 < 0.005:
                color, status = "green", "Excelente"
            elif sigma0 < 0.02:
                color, status = "orange", "Aceptable"
            else:
                color, status = "red", "Pobre"
                
            st.markdown(f"**Varianza a Posteriori ($\\sigma_0$):** :{color}[**{sigma0:.4f} m** ({status})]")
            st.markdown("---")

    # 3. Residuals Table
    if "residuals" in data:
        st.subheader("Cuadrícula de Residuales")
        residuals = data["residuals"]
        if isinstance(residuals, list) and len(residuals) > 0:
             df = pd.DataFrame(residuals)
             
             # Highlight outliers style
             def highlight_outliers(row):
                 is_outlier = row.get("outlier_horizontal", False) or row.get("outlier_vertical", False)
                 color = 'background-color: #ffcccc; color: #900' if is_outlier else ''
                 return [color] * len(row)
                 
             display_df = df.copy()
             if "outlier_horizontal" in display_df.columns:
                 display_df["outlier_horizontal"] = display_df["outlier_horizontal"].apply(lambda x: "⚠️ Outlier" if x else "✅")
             if "outlier_vertical" in display_df.columns:
                 display_df["outlier_vertical"] = display_df["outlier_vertical"].apply(lambda x: "⚠️ Outlier" if x else "✅")
                 
             display_df.rename(columns={"dE": "dE (m)", "dN": "dN (m)", "dH": "dH (m)"}, inplace=True)
             st.dataframe(display_df.style.apply(highlight_outliers, axis=1), use_container_width=True)
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
