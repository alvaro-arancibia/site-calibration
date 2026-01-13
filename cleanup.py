import os
from pathlib import Path

def cleanup():
    # 1. Archivos que sabemos que son obsoletos
    obsolete_files = [
        "src/sitecal/calibration.py", 
        "src/sitecal/tbc_html.py",
        "src/sitecal/tbc_default_tm.py",
        "src/sitecal/commands/local2global.py",
        "src/sitecal/csv_handler.py",
        "src/sitecal/csv_repository.py",
    ]
    
    # 2. Buscar archivos vacíos en src
    for path in Path("src").rglob("*.py"):
        if path.is_file() and path.stat().st_size == 0:
            print(f"Eliminando archivo vacío: {path}")
            path.unlink()

    for f in obsolete_files:
        p = Path(f)
        if p.exists():
            print(f"Eliminando archivo obsoleto: {f}")
            p.unlink()
            
    # 3. Try to remove the commands directory if it is empty
    try:
        Path("src/sitecal/commands").rmdir()
        print("Eliminando el directorio 'src/sitecal/commands'.")
    except OSError as e:
        print(f"No se pudo eliminar el directorio 'src/sitecal/commands': {e}")


if __name__ == "__main__":
    cleanup()
