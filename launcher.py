#!/usr/bin/env python
import subprocess
import sys
import os
import time
import webbrowser
import platform

def open_url_once(url: str):
    # Usa webbrowser solo una vez, sin fallback
    try:
        webbrowser.open_new(url)
        print(f"DEBUG: Intento único de abrir navegador en {url}")
    except Exception as e:
        print(f"ERROR: No se pudo abrir el navegador: {e}")

if __name__ == "__main__":
    print("DEBUG: Iniciando launcher.py")
    base_path = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    app_path = os.path.join(base_path, "app.py")
    print(f"DEBUG: Ruta de app.py -> {app_path}")

    cmd = [sys.executable, "-m", "streamlit", "run", app_path]
    print(f"DEBUG: Comando -> {cmd}")

    proc = subprocess.Popen(cmd)
    print("DEBUG: Proceso Streamlit lanzado, esperando a que arranque...")

    # Espera suficiente para que Streamlit arranque
    time.sleep(5)

    url = "http://localhost:8501"
    open_url_once(url)

    print("DEBUG: Launcher terminado. Streamlit sigue corriendo en background.")
    proc.wait()
    print("DEBUG: Streamlit finalizó")
