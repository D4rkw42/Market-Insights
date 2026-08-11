# Compilação do projeto client com Nuitka

import os, subprocess

APP_EXECUTABLE_NAME = "Market-Insights"

PROJECT_DIR = "src/market_insights"
BUILD_DIR = "build/project"

COMPILED_PROJECT = "market_insights.dist"

COMPILE_COMMAND = f"poetry run python -m nuitka --msvc=latest --standalone --follow-imports --enable-plugin=pyside6 --output-dir={BUILD_DIR} --output-filename={APP_EXECUTABLE_NAME}.exe {PROJECT_DIR}"

# Compila o projeto Client com Nuitka

def export_app():
    print(f"[scripts.build_app] Compiling {APP_EXECUTABLE_NAME}...")
    print("[scripts.build_app]: " + COMPILE_COMMAND)

    # Verifica se a pasta de build existe

    if not os.path.exists(BUILD_DIR):
        os.makedirs(BUILD_DIR)

    # Execução de comando

    os.system(COMPILE_COMMAND)

    print("[scripts.build_app] Compiled successfully!")

# Executa o projeto Client compilado

def run_app():
    print(f"[scripts.build_app]: Starting {APP_EXECUTABLE_NAME}.exe...")
    subprocess.call([f"./{BUILD_DIR}/{COMPILED_PROJECT}/Market-Insights"])

if __name__ == "__main__":
    export_app()
    run_app()
