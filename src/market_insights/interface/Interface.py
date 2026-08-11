# Interface principal do projeto

from PySide6 import QtWidgets

class Interface(QtWidgets.QVBoxLayout):
    def __init__(self, parent: QtWidgets.QWidget):
        super().__init__(parent)
