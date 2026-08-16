import os

from dotenv import load_dotenv

# Carrega os dados do .env local
load_dotenv()

# Definição do modo de operação

OPERATION_MODE = os.getenv("OPERATION_MODE")

# Nome exibido na janela

APP_NAME = "Market Insights"

# Dimensões mínimas da janela

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
