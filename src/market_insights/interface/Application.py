# Aplicação do projeto

from PySide6 import QtWidgets, QtGui

from .Interface import Interface

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

    def Init(self):
        pass

    # Finalização do app

    def Quit(self):
        pass

    # Execução de funções internas

    def Update(self, dt: float):
        pass
