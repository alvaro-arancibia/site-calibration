# Documentación de Calibración de Sitio (`sitecal`)

`sitecal` es una herramienta de línea de comandos diseñada para realizar calibraciones de sitio compatibles con Trimble Business Center (TBC). Permite transformar coordenadas globales geodéticas (Latitud, Longitud) en el datum WGS84 a un sistema de coordenadas plano local mediante diversos métodos de proyección y un motor de ajuste de similitud 2D.

## Flujo de Trabajo

El proceso típico de calibración en `sitecal` sigue estos pasos:

1.  **Entrada de Datos**: Se requieren dos archivos CSV:
    *   **Global**: Contiene puntos con coordenadas geodéticas WGS84 (`Point`, `Lat`, `Lon`, `h`).
    *   **Local**: Contiene los mismos puntos en el sistema de coordenadas de destino (`Point`, `Easting`, `Northing`, `h_local`).
2.  **Proyección**: Las coordenadas globales se proyectan a un plano intermedio utilizando el método seleccionado (TBC Default, UTM o LTM).
3.  **Ajuste (Entrenamiento)**: Se calculan los parámetros de transformación de Similitud 2D comparando las coordenadas proyectadas con las locales.
4.  **Generación de Reporte**: Se crea un archivo Markdown detallando los parámetros obtenidos (traslaciones, rotación y escala) y los residuales en cada punto.
5.  **Transformación (Opcional)**: Se pueden aplicar los parámetros calculados para transformar otros puntos globales al sistema local.

---

## Métodos de Proyección

La aplicación soporta los siguientes métodos para proyectar coordenadas geodéticas:

### 1. TBC Default
Este método emula el comportamiento por defecto de TBC cuando no se define un sistema de coordenadas.
*   **Origen**: Utiliza el primer punto del archivo global como origen (0,0).
*   **Proyección**: Transverse Mercator local con factor de escala 1.0.

### 2. UTM (Universal Transverse Mercator)
Determina automáticamente la zona UTM basándose en la longitud media de los puntos de entrada.
*   **Detección de Hemisferio**: Identifica si los puntos están en el hemisferio norte o sur para asignar el código EPSG correcto.

### 3. LTM (Local Transverse Mercator)
Permite definir una proyección personalizada mediante parámetros específicos:
*   Meridiano Central
*   Latitud de Origen
*   Falso Este / Falso Norte
*   Factor de Escala

---

## Motores de Calibración

### Similitud 2D (2D Similarity)
Es el motor principal de calibración para este MVP. Realiza una transformación de 4 parámetros en el plano:
*   2 Traslaciones (Norte, Este)
*   1 Rotación
*   1 Factor de escala

Este ajuste se realiza mediante mínimos cuadrados, utilizando una técnica de centrado de coordenadas para garantizar estabilidad numérica máxima. Es ideal para ajustar levantamientos GNSS a redes locales preexistentes.

---

## Uso de la CLI

El comando principal es `local2global`. A continuación se muestra un ejemplo básico:

```bash
sitecal local2global \
  --global-csv data/global_points.csv \
  --local-csv data/local_points.csv \
  --method tbc \
  --output-report mi_reporte.md
```

### Parámetros para LTM
Si utilizas el método `ltm`, debes proporcionar los parámetros adicionales:
```bash
sitecal local2global \
  --global-csv data/global_points.csv \
  --local-csv data/local_points.csv \
  --method ltm \
  --central-meridian -70.5 \
  --latitude-of-origin -33.4 \
  --false-easting 500000 \
  --false-northing 10000000 \
  --scale-factor 1.0
```

---

## Resultados

### Reporte de Calibración
El archivo generado (`calibration_report.md` por defecto) incluye:
- Parámetros de la transformación calculada.
- Listado de puntos utilizados con sus residuales (dE, dN).
- Error medio cuadrático (RMS) del ajuste.

### Coordenadas Transformadas
Si se especifica `--output-csv`, la aplicación generará un archivo con las coordenadas proyectadas y ajustadas para todos los puntos de entrada.
