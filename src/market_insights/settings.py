import os

from dotenv import load_dotenv

# Carrega os dados do .env local
load_dotenv()

# Definição do modo de operação

OPERATION_MODE = os.getenv("OPERATION_MODE")
