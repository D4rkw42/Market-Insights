# Definições importantes para conexão à API da Binance

import os

from market_insights.settings import *
from market_insights.utils.public import *

class Binance:
    def __init__(self):
        pass

    def init(self):
        self.api_url = os.getenv("API_URL") if app_is_prod_mode() else os.getenv("TESTNET_API_URL")
        self.api_key = os.getenv("API_KEY") if app_is_prod_mode() else os.getenv("TESTNET_API_KEY")
        self.secret_key = os.getenv("SECRET_KEY") if app_is_prod_mode() else os.getenv("TESTNET_SECRET_KEY")

        if not (self.api_url and self.api_key and self.secret_key):
            raise RuntimeError("Couldn't load ENV vars about Binance connection")
