# Instrucciones de Instalación PWA (Site Calibration Pro)

Esta aplicación ha sido empaquetada como una Progressive Web App (PWA) de archivo único. Funciona completamente offline y ejecuta el motor de calibración (Python/NumPy) directamente en el navegador de tu tablet o dispositivo móvil.

## Opción 1: Ejecutar Localmente

Si tienes Python instalado en tu computadora y quieres probarlo antes de subirlo:

1. Navega a la carpeta `dist`:

   ```bash
   cd dist
   ```

2. Inicia un servidor simple:

   ```bash
   python3 -m http.server 8000
   ```

3. Abre tu navegador en `http://localhost:8000`.

## Opción 2: Instalar en Tablet (GitHub Pages)

Para instalarlo en tu tablet Android o iPad:

1. **Subir a GitHub Pages**:
   - Sube el contenido de la carpeta `dist/` a un repositorio de GitHub (en una rama `gh-pages` o configurando `docs/` como fuente).
   - Asegúrate de que `index.html` sea el archivo principal.

2. **Acceder desde la Tablet**:
   - Abre la URL de tu GitHub Page (ej: `https://tu-usuario.github.io/tu-repo/`) en Chrome (Android) o Safari (iOS).

3. **Instalar (Añadir a Inicio)**:
   - **Android (Chrome)**: Toca el menú de tres puntos -> "Instalar aplicación" o "Añadir a pantalla de inicio".
   - **iOS (Safari)**: Toca el botón "Compartir" (flecha saliendo de caja) -> "Añadir a pantalla de inicio".

## Notas Técnicas

- La primera carga puede tomar unos segundos mientras se descarga el entorno de Python (Pyodide). Las siguientes cargas serán instantáneas (cache).
- No se envían datos a ningún servidor. Todo el procesamiento ocurre en la memoria de tu dispositivo.
