# animation-frame-toolkit

Herramientas para limpiar y extraer personajes 2D con fondo blanco
para integrarlos en imágenes reales.

Contenido principal
- `scripts/cartoon_frame_cleaner.py`: herramienta CLI existente (v8).
- `animation_frame_toolkit/`: paquete Python con accesos directos.

Instalación
1. Crear y activar un virtualenv (recomendado).
2. Instalar dependencias:

```bash
pip install -r requirements.txt
```

Uso rápido
- Procesar una sola imagen:

```bash
python3 scripts/cartoon_frame_cleaner.py "media/Gato 01/EXPORT_frames/input.png" out.png
```

- Procesar una carpeta de imágenes (salida en carpeta):

```bash
python3 scripts/cartoon_frame_cleaner.py "media/Gato 01/EXPORT_frames/" outputs/
```

Interfaz de paquete
Puedes usarlo como módulo:

```bash
python -m animation_frame_toolkit "media/Gato 01/EXPORT_frames/" outputs/
```

Estructura propuesta

- `media/` — tus imágenes y referencias (sin cambios).
- `scripts/` — utilidades ejecutables (CLI principal).
- `animation_frame_toolkit/` — paquete Python (API mínima).
- `requirements.txt` — dependencias.
- `README.md`, `.gitignore`.

Contribuciones
Si quieres que automatice tests, empaquetado o CI, dímelo y lo añado.
