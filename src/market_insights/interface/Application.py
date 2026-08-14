# Aplicação do projeto

from PySide6 import QtWidgets, QtGui

from market_insights.settings import *
from market_insights.globals import *

from market_insights.interface import Interface

from datetime import datetime

# Defnições do aplicativo

# Dimennsões mínimas da janela

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

APP_NAME = "Market Insights" # Nome exibido na janela

class Application(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        # Configurações

        self.setWindowTitle(APP_NAME)

        self.setMinimumSize(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.resize(WINDOW_WIDTH, WINDOW_HEIGHT)

        # Ícone da janela

        icon = QtGui.QIcon("assets/favicon.ico")
        self.setWindowIcon(icon)

        # Definindo a interface

        container = QtWidgets.QWidget()
        self.setCentralWidget(container)

        Interface(container)

    # Inicialização do app

    def init(self):
        if OPERATION_MODE == "develop":
            print("[LOG] Running app in development mode.")

        # Inicializa conexão com os endpoints da Binance

        binance.init()

    # Finalização do app

    def quit(self):
        pass

    # Execução de funções internas

    def load(self, dt: float):
        pass
