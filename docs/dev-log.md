# sitecal · Dev Log

Registro de bugs encontrados, causa raíz, solución y lecciones aprendidas.

---

## 2026-02-22 — Sprint 3: Fix transformación bidireccional

**Contexto:** Después de implementar el flujo Local↔Global y la serialización `.sitecal`, se detectaron deltas métricos en tests round-trip. Claude Code (Opus 4.6) hizo el diagnóstico completo leyendo `app.py` y `calibration_engine.py`.

---

### Bug 1 — Crítico: proyección faltante en Global→Local

**Archivo:** `src/sitecal/ui/app.py` (~línea 280)

**Causa raíz:** Al cargar un CSV con coordenadas globales (Lat/Lon en grados) y llamar a `transform()`, no se ejecutaba el paso `projection.project()` previo. El motor recibía valores en grados (≈ −33, −70) como si fueran metros proyectados, mientras que los centroides internos están en cientos de miles de metros. Resultado completamente incorrecto.

Durante el entrenamiento el paso sí existía (líneas 142-143), pero la sección "Transformar Puntos" lo omitió.

**Solución:** Instanciar `ProjectionFactory` con los parámetros guardados en el `.sitecal` y ejecutar `projection.project()` antes de llamar a `loaded_engine.transform()`. Se renombraron columnas intermedias a `Longitude`/`Latitude` para evitar confusión semántica.

**Lección:** Cuando se replica un flujo existente en una nueva sección de UI, verificar explícitamente que todos los pasos del pipeline estén presentes — especialmente transformaciones "invisibles" como la proyección cartográfica.

---

### Bug 2 — AttributeError en else-branch de transform_inverse()

**Archivo:** `src/sitecal/core/calibration_engine.py` (~línea 265)

**Causa raíz:**
```python
# Antes — crash si "Easting" no existe
E_in = df.get("Easting", np.zeros(len(df))).values
```
`pd.DataFrame.get(key, default)` retorna el `default` cuando la clave no existe. Si `"Easting"` no está en el DataFrame, devuelve un `np.ndarray` — que no tiene atributo `.values`. En scripts funciona porque la columna siempre existe; desde la UI puede no estar si el CSV usa nombres distintos.

**Solución:**
```python
# Después — acceso seguro
E_in = df["Easting"].values if "Easting" in df.columns else np.zeros(len(df))
```

**Lección:** `df.get(key, ndarray_default).values` es un antipatrón silencioso. El fallback de `DataFrame.get()` no garantiza tener el mismo contrato de interfaz que una Series. Usar siempre `key in df.columns` para bifurcar explícitamente.

---

### Bug 3 — Columnas fantasma tras rename en Local→Global

**Archivo:** `src/sitecal/ui/app.py` (~línea 284)

**Causa raíz:**
```python
# Antes — columnas sobrantes del CSV original permanecen
df_ready = trans_df.rename(columns={t_id: "Point", t_x: "Easting_local", ...})
```
`rename()` no elimina columnas: todas las del CSV original pasan al motor. Si el CSV contiene una columna que ya se llama `"Easting_local"` (distinta de la seleccionada), el DataFrame tiene dos columnas con el mismo nombre. `df["Easting_local"].values` retorna un DataFrame en lugar de una Series, y los cálculos matriciales fallan silenciosamente o producen resultados incorrectos.

**Solución:**
```python
# Después — solo las 4 columnas necesarias
df_ready = trans_df.rename(columns={...})[["Point", "Easting_local", "Northing_local", "Elevation"]]
```

**Lección:** Después de un `rename()` para preparar datos de entrada a un motor, siempre hacer una selección explícita de columnas. Esto garantiza el contrato de interfaz y evita contaminación por columnas del CSV del usuario.

---

### Validación post-fix
- 29/29 tests Golden State: ✅ sin regresiones
- Commit: `62be608` — branch `refactor`

---

## Historial de commits relevantes

| Fecha | Commit | Descripción |
|-------|--------|-------------|
| 2026-02-22 | `62be608` | Fix: 3 bugs transformación bidireccional |
| 2026-02-22 | anterior | Sprint 3: serialización .sitecal + flujo bidireccional |
| 2026-02-22 | anterior | Fix: transform_inverse() completo — deproyección PyProj |
| 2026-02-22 | anterior | Sprint 2: integración UI — WLS, outliers, alertas, semáforo σ₀ |
| 2026-02-22 | anterior | Fix: tE y tN perdidos en desacople arquitectónico |
| 2026-02-22 | anterior | Sprint 2: desacople arquitectónico Pydantic + core NumPy puro |
| 2026-02-22 | anterior | Sprint 2: WLS con centroides ponderados y fallback OLS |
| 2026-02-22 | anterior | Sprint 2: detección outliers Test de Baarda |
| 2026-02-22 | anterior | Sprint 2: alertas extrapolación ConvexHull en transform() |