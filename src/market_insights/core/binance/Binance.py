# Definições importantes para conexão à API da Binance

import os

from binance_sdk_spot.spot import Spot
from binance_common.configuration import ConfigurationRestAPI

from market_insights.settings import *
from market_insights.utils.public import *

class Binance:
    def __init__(self):
        pass

    def init(self):
        self.api_url = os.getenv("API_URL") if app_is_prod_mode() else os.getenv("TESTNET_API_URL")
        self.api_key = os.getenv("API_KEY") if app_is_prod_mode() else os.getenv("TESTNET_API_KEY")
        self.secret_key = os.getenv("SECRET_KEY") if app_is_prod_mode() else os.getenv("TESTNET_SECRET_KEY")

        print(self.api_key, self.secret_key)

        rest_api_configuration = ConfigurationRestAPI(
            api_key = self.api_key,
            private_key = self.secret_key,
            base_path = self.api_url
        )

        self.client = Spot(config_rest_api = rest_api_configuration)

    # Dados da conta vinculada
    def get_account(self):
        return self.client.rest_api.get_account()
