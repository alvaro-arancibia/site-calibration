import streamlit as st
import pandas as pd
import io
import numpy as np

# Core Imports for Offline Processing
from sitecal.core.calibration_engine import Similarity2D
from sitecal.core.projections import ProjectionFactory
from sitecal.infrastructure.reports import generate_markdown_report

TOTAL_STEPS = 4
STEP_LABELS = [
    "1. Carga de Archivos",
    "2. Mapeo y Método",
    "3. Preview y Validación",
    "4. Resultados y Transformación",
]


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


def _init_state():
    """Initialize session_state defaults once."""
    defaults = {
        "step": 1,
        "global_df": None,
        "local_df": None,
        "use_wls": False,
        "col_map_g": {},
        "col_map_l": {},
        "method": "Default",
        "params": {},
        "merged_df": None,
        "df_g_ready": None,
        "df_g_proj": None,
        "df_l_ready": None,
        "cal_engine": None,
        "cal_result": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _go(step: int):
    st.session_state["step"] = step


# ── Progress bar ──────────────────────────────────────────────
def _render_progress():
    step = st.session_state["step"]
    st.progress(step / TOTAL_STEPS, text=f"Paso {step} de {TOTAL_STEPS} — {STEP_LABELS[step - 1]}")


# ── Step 1: File Upload ──────────────────────────────────────
def _step_upload():
    st.header("Paso 1 — Carga de Archivos CSV")

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

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Coordenadas Globales (CSV)")
        global_file = st.file_uploader("Subir CSV Global", type=["csv"], key="global")
        if global_file:
            has_header_g = st.checkbox("Tiene encabezados", value=True, key="header_g")
            gdf = pd.read_csv(global_file, header=0 if has_header_g else None)
            st.session_state["global_df"] = gdf
            st.dataframe(gdf.head(), use_container_width=True)
        else:
            st.session_state["global_df"] = None

    with col2:
        st.subheader("Coordenadas Locales (CSV)")
        local_file = st.file_uploader("Subir CSV Local", type=["csv"], key="local")
        if local_file:
            has_header_l = st.checkbox("Tiene encabezados", value=True, key="header_l")
            ldf = pd.read_csv(local_file, header=0 if has_header_l else None)

            use_wls = st.checkbox("Ingresar precisión por punto (WLS)", value=False, key="use_wls_cb")
            st.session_state["use_wls"] = use_wls
            if use_wls:
                if "sigma" not in ldf.columns:
                    ldf["sigma"] = 1.0
                st.caption("Edita la columna 'sigma' para aplicar pesos personalizados (WLS):")
                ldf = st.data_editor(ldf, num_rows="dynamic", key="local_editor")
            else:
                st.dataframe(ldf.head(), use_container_width=True)

            st.session_state["local_df"] = ldf
        else:
            st.session_state["local_df"] = None

    # Navigation
    st.markdown("---")
    both_ready = st.session_state["global_df"] is not None and st.session_state["local_df"] is not None
    if not both_ready:
        st.info("Sube ambos archivos CSV para continuar.")
    st.button("Siguiente →", disabled=not both_ready, on_click=_go, args=(2,), type="primary", use_container_width=True)


# ── Step 2: Column Mapping + Method ──────────────────────────
def _step_mapping():
    st.header("Paso 2 — Mapeo de Columnas y Método")

    global_df = st.session_state["global_df"]
    local_df = st.session_state["local_df"]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Columnas Globales")
        st.dataframe(global_df.head(3), use_container_width=True)
        cols_g = global_df.columns.tolist()
        g_point = st.selectbox("Point (ID)", cols_g, index=0, key="g_pt")
        g_lat = st.selectbox("Latitude", cols_g, index=1 if len(cols_g) > 1 else 0, key="g_lat")
        g_lon = st.selectbox("Longitude", cols_g, index=2 if len(cols_g) > 2 else 0, key="g_lon")
        g_h = st.selectbox("Ellipsoidal Height", cols_g, index=3 if len(cols_g) > 3 else 0, key="g_h")

    with col2:
        st.subheader("Columnas Locales")
        st.dataframe(local_df.head(3), use_container_width=True)
        cols_l = local_df.columns.tolist()
        l_point = st.selectbox("Point (ID)", cols_l, index=0, key="l_pt")
        l_e = st.selectbox("Easting", cols_l, index=1 if len(cols_l) > 1 else 0, key="l_e")
        l_n = st.selectbox("Northing", cols_l, index=2 if len(cols_l) > 2 else 0, key="l_n")
        l_z = st.selectbox("Elevation", cols_l, index=3 if len(cols_l) > 3 else 0, key="l_z")

        st.markdown("##### Geometría Local")
        try:
            chart_data = local_df.rename(columns={l_e: "Easting", l_n: "Northing"})
            st.scatter_chart(chart_data, x="Easting", y="Northing", color="#FF4B4B")
        except Exception:
            st.caption("No se pudo generar la previsualización gráfica.")

    # Store column mapping
    st.session_state["col_map_g"] = {"Point": g_point, "Latitude": g_lat, "Longitude": g_lon, "EllipsoidalHeight": g_h}
    st.session_state["col_map_l"] = {"Point": l_point, "Easting": l_e, "Northing": l_n, "Elevation": l_z}

    # Method selection
    st.subheader("Método y Parámetros")
    col_method, col_params = st.columns([1, 3])

    with col_method:
        method = st.selectbox("Seleccionar Método", ["Default", "LTM", "EPSG"])
    st.session_state["method"] = method

    params = {}
    if method == "LTM":
        with col_params:
            c1, c2, c3, c4 = st.columns(4)
            with c1: params["central_meridian"] = st.number_input("Meridiano Central", value=-72.0)
            with c2: params["scale_factor"] = st.number_input("Factor de Escala", value=0.9996, format="%.6f")
            with c3: params["false_easting"] = st.number_input("Falso Este", value=500000.0)
            with c4: params["false_northing"] = st.number_input("Falso Norte", value=10000000.0)
    elif method == "EPSG":
        with col_params:
            params["epsg_code"] = st.number_input(
                "Código EPSG",
                value=32719,
                min_value=1024,
                max_value=32767,
                step=1,
                help="Ej: 32719 = WGS 84 / UTM zone 19S, 32718 = UTM 18S, 5361 = PSAD56 / Peru West Zone",
            )
    st.session_state["params"] = params

    # Navigation
    st.markdown("---")
    c_back, c_next = st.columns(2)
    with c_back:
        st.button("← Atrás", on_click=_go, args=(1,), use_container_width=True)
    with c_next:
        st.button("Siguiente →", on_click=_go, args=(3,), type="primary", use_container_width=True)


# ── Step 3: Preview + Validations ────────────────────────────
def _step_preview():
    st.header("Paso 3 — Preview y Validación")

    global_df = st.session_state["global_df"]
    local_df = st.session_state["local_df"]
    cmap_g = st.session_state["col_map_g"]
    cmap_l = st.session_state["col_map_l"]
    method = st.session_state["method"]
    params = st.session_state["params"]
    use_wls = st.session_state["use_wls"]

    try:
        # Standardize global
        df_g_ready = global_df.rename(columns={
            cmap_g["Point"]: "Point",
            cmap_g["Latitude"]: "Latitude",
            cmap_g["Longitude"]: "Longitude",
            cmap_g["EllipsoidalHeight"]: "EllipsoidalHeight",
        })[["Point", "Latitude", "Longitude", "EllipsoidalHeight"]]
        df_g_ready["Point"] = df_g_ready["Point"].astype(str)

        # Standardize local
        cols_to_keep_l = ["Point", "Easting", "Northing", "Elevation"]
        if "sigma" in local_df.columns and use_wls:
            cols_to_keep_l.append("sigma")

        df_l_ready = local_df.rename(columns={
            cmap_l["Point"]: "Point",
            cmap_l["Easting"]: "Easting",
            cmap_l["Northing"]: "Northing",
            cmap_l["Elevation"]: "Elevation",
        })[cols_to_keep_l]
        df_l_ready["Point"] = df_l_ready["Point"].astype(str)

        # Project
        proj_params = {k: v for k, v in params.items()}
        projection = ProjectionFactory.create(method.lower(), **proj_params)
        df_g_proj = projection.project(df_g_ready)

        # Merge
        merged_df = pd.merge(df_l_ready, df_g_proj, on="Point", suffixes=('_local', '_global'))

        # Store for step 4
        st.session_state["df_g_ready"] = df_g_ready
        st.session_state["df_g_proj"] = df_g_proj
        st.session_state["df_l_ready"] = df_l_ready
        st.session_state["merged_df"] = merged_df

    except Exception as e:
        st.error(f"Error preparando datos: {e}")
        st.button("← Atrás", on_click=_go, args=(2,), use_container_width=True)
        return

    # Show common points
    n = len(merged_df)
    st.subheader(f"Puntos comunes detectados: {n}")
    st.dataframe(merged_df, use_container_width=True)

    # Validations
    errors = []
    if n < 3:
        errors.append(f"Solo se encontraron **{n}** puntos comunes. Se requieren mínimo 3.")
    if n >= 3 and validate_collinearity(merged_df):
        errors.append("Los puntos son colineales o geográficamente muy cercanos. Geometría inestable.")

    if errors:
        for e in errors:
            st.error(e)
        can_proceed = False
    else:
        st.success(f"{n} puntos comunes válidos. Geometría aceptable.")
        can_proceed = True

    # Navigation
    st.markdown("---")
    c_back, c_next = st.columns(2)
    with c_back:
        st.button("← Atrás", on_click=_go, args=(2,), use_container_width=True)
    with c_next:
        st.button("Calcular Calibración →", disabled=not can_proceed, on_click=_go, args=(4,), type="primary", use_container_width=True)


# ── Step 4: Calibration + Results + Transform ────────────────
def _step_results():
    st.header("Paso 4 — Resultados de Calibración")

    df_g_ready = st.session_state["df_g_ready"]
    df_g_proj = st.session_state["df_g_proj"]
    df_l_ready = st.session_state["df_l_ready"]
    method = st.session_state["method"]
    params = st.session_state["params"]

    # Run calibration (once per visit — cached in session_state)
    if st.session_state.get("cal_result") is None:
        with st.spinner("Procesando localmente..."):
            try:
                import warnings
                engine = Similarity2D()

                # Capture projection params for serialization
                engine.proj_method = method.lower()
                adj_params = params.copy()
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
                elif engine.proj_method == "epsg":
                    adj_params["epsg_code"] = int(params.get("epsg_code", 32719))
                engine.proj_params = adj_params

                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    engine.train(df_l_ready, df_g_proj)
                    for warn in w:
                        if "Extrapolación detectada" in str(warn.message):
                            st.warning(str(warn.message), icon="⚠️")

                # Build result
                residuals = []
                for _, row in engine.residuals.iterrows():
                    res_dict = {
                        "Point": str(row["Point"]),
                        "dE": float(row["dE"]),
                        "dN": float(row["dN"]),
                        "dH": float(row["dH"]),
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
                        "vertical": engine.vertical_params,
                    },
                    "residuals": residuals,
                    "report": report_text,
                }

                st.session_state["cal_engine"] = engine
                st.session_state["cal_result"] = result_data

            except Exception as e:
                st.error(f"Error Interno: {str(e)}")
                st.button("← Atrás", on_click=_go, args=(3,), use_container_width=True)
                return

    result_data = st.session_state["cal_result"]
    engine = st.session_state["cal_engine"]

    display_results(result_data)

    st.download_button(
        label="💾 Descargar Calibración (.sitecal)",
        data=engine.save(),
        file_name="calibracion.sitecal",
        mime="application/json",
        use_container_width=True,
    )

    # ── Transform section ─────────────────────────────────────
    st.markdown("---")
    st.header("Transformar Puntos")
    st.markdown("Aplica una calibración existente a un nuevo conjunto de puntos.")

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
        st.subheader("Puntos a Transformar")
        trans_file = st.file_uploader("Subir CSV de Puntos", type=["csv"], key="trans_points")

        if trans_file:
            trans_df = pd.read_csv(trans_file)
            st.dataframe(trans_df.head(), use_container_width=True)

            direction = st.radio("Dirección de Transformación", ["Global -> Local", "Local -> Global"])

            tc = trans_df.columns.tolist()
            t_id = st.selectbox("Point (ID)", tc, index=0, key="t_id")
            if direction == "Global -> Local":
                t_x = st.selectbox("Easting Global / Longitude", tc, index=1 if len(tc) > 1 else 0, key="t_g_e")
                t_y = st.selectbox("Northing Global / Latitude", tc, index=2 if len(tc) > 2 else 0, key="t_g_n")
                t_z = st.selectbox("Ellipsoidal Height / h", tc, index=3 if len(tc) > 3 else 0, key="t_g_h")
            else:
                t_x = st.selectbox("Easting (Local)", tc, index=1 if len(tc) > 1 else 0, key="t_l_e")
                t_y = st.selectbox("Northing (Local)", tc, index=2 if len(tc) > 2 else 0, key="t_l_n")
                t_z = st.selectbox("Elevation (Local)", tc, index=3 if len(tc) > 3 else 0, key="t_l_h")

            if st.button("Aplicar Transformación", type="primary", key="btn_trans"):
                with st.spinner("Transformando..."):
                    try:
                        if direction == "Global -> Local":
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
                            mime="text/csv",
                        )
                    except Exception as e:
                        st.error(f"Error transformando puntos: {str(e)}")

    # Navigation
    st.markdown("---")
    c_back, c_new = st.columns(2)
    with c_back:
        st.button("← Atrás (Preview)", on_click=_go, args=(3,), use_container_width=True)
    with c_new:
        if st.button("Nueva Calibración", use_container_width=True):
            for k in ["cal_result", "cal_engine", "merged_df", "df_g_ready", "df_g_proj", "df_l_ready"]:
                st.session_state[k] = None
            _go(1)


# ── Display results (unchanged logic) ────────────────────────
def display_results(data):
    # 1. Calculated Parameters
    if "parameters" in data:
        p = data["parameters"]

        if "horizontal" in p and p["horizontal"]:
            st.subheader("🏗️ Ajuste Horizontal (2D)")
            hp = p["horizontal"]
            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("Factor Escala (a)", f"{hp['a']:.7f}")
            with c2: st.metric("Rotación (b)", f"{hp['b']:.7f}")
            with c3: st.metric("Traslasión Este", f"{hp['tE']:.3f} m")
            with c4: st.metric("Traslasión Norte", f"{hp['tN']:.3f} m")

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

        sigma0_sq = hp.get("sigma0_sq_h")
        residuals_list = data["residuals"]

        if residuals_list:
            if sigma0_sq is not None:
                sigma0 = np.sqrt(sigma0_sq)
            else:
                df_res_tmp = pd.DataFrame(residuals_list)
                dof = max(1, 2 * len(df_res_tmp) - 4)
                v_sq = df_res_tmp["dE"]**2 + df_res_tmp["dN"]**2
                sigma0 = np.sqrt(v_sq.sum() / dof)

            st.subheader("🎯 Calidad del Ajuste (Horizontal)")
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

    # 4. Full Report
    if "report" in data:
        st.subheader("Reporte Completo de Calibración")
        st.markdown(data["report"])
    elif "markdown_report" in data:
        st.subheader("Reporte Completo de Calibración")
        st.markdown(data["markdown_report"])


# ── Main ──────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Site Calibration (Offline)", page_icon="🛰️", layout="wide")
    _init_state()

    st.title("Site Calibration Tool (Monolith)")
    st.markdown("Cálculo local y seguro. No requiere conexión a internet.")

    _render_progress()

    step = st.session_state["step"]
    if step == 1:
        _step_upload()
    elif step == 2:
        _step_mapping()
    elif step == 3:
        _step_preview()
    elif step == 4:
        _step_results()


if __name__ == "__main__":
    main()
