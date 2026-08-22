# An application built upon a LSTM neural network that creates insights about financial market for buying or selling operations
# D4rkw42

import sys

from PySide6 import QtWidgets
from PySide6.QtCore import QTimer, QElapsedTimer

from market_insights.settings import *
from market_insights.globals import *

from market_insights.interface import Application

# Definições do aplicativo

APP_UPDATE_COUNTDOWN = 500 # Tempo de execução (ms)

def main():
    QtApp = QtWidgets.QApplication(sys.argv)
    application = Application()

    # Inicialização do aplicativo

    application.init()

    # Definição do loop principal

    clock = QElapsedTimer()
    timer = QTimer()

    # Callback do timeout, passando dt em milisegundos

    def timeout_callback():
        dt = clock.restart() * 0.001
        application.load(dt)

    timer.setInterval(APP_UPDATE_COUNTDOWN)
    timer.timeout.connect(timeout_callback)

    clock.start()
    timer.start()

    # Starting

    application.show()
    QtApp.exec()

    # Finalização do aplicativo

    application.quit()

if __name__ == "__main__":
    main()
