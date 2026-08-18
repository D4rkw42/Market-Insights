# Definições importantes para conexão à API da Binance

import os

from requests import get, post
from typing import Any

from datetime import datetime

from market_insights.settings import *
from market_insights.utils.public import *

from market_insights.utils.math import ScientificNotation

# Binance endpoints

BINANCE_EXCHANGE_INFO_ENDPOINT = "/api/v3/exchangeInfo" # Informações sobre pares de moedas
BINANCE_AVERAGE_PRICE_ENDPOINT = "/api/v3/avgPrice"     # Informações sobre o preço médio do ativo
BINANCE_KLINES_ENDPOINT = "/api/v3/klines"              # Informações acerca de candlesticks

class Binance:
    def __init__(self):
        pass

    # Carrega as configurações de ambiente da Binance

    def init(self):
        self.__api_url = os.getenv("API_URL") if app_is_prod_mode() else os.getenv("TESTNET_API_URL")
        self.__api_key = os.getenv("API_KEY") if app_is_prod_mode() else os.getenv("TESTNET_API_KEY")
        self.__secret_key = os.getenv("SECRET_KEY") if app_is_prod_mode() else os.getenv("TESTNET_SECRET_KEY")

        if not (self.__api_url and self.__api_key and self.__secret_key):
            raise RuntimeError("Couldn't load ENV vars about Binance connection")


    # Chamada dinâmica de API da Binance

    def __call_api_endpoint(self, endpoint: str, parameters: dict[str, Any] = {}, method: str = "get") -> BinanceEndpointResponse:
        if not method in ["get", "post"]:
            return BinanceEndpointResponse(-1)

        http_method = get if method == "get" else post
        url = self.__api_url + endpoint # type: ignore

        response = http_method(url, parameters)

        response_object = {}

        try:
            response_object = response.json()
        except:
            pass

        return BinanceEndpointResponse(
            response.status_code,
            response.reason,
            response_object
        )


    # Obtém as informações acerca de um par de moedas

    def __get_exchange_info(self, symbol: str) -> BinanceEndpointResponse:
        return self.__call_api_endpoint(BINANCE_EXCHANGE_INFO_ENDPOINT, { "symbol": symbol })

    # Obtém o preço médio de um par no momento atual

    def __get_average_price(self, symbol: str) -> BinanceEndpointResponse:
        return self.__call_api_endpoint(BINANCE_AVERAGE_PRICE_ENDPOINT, { "symbol": symbol })

    # Obtém as informações acerca dos candlesticks de um par de moeda
    
    def __get_klines(self, symbol: str, interval: str, start_time: int, end_time: int, limit: int = 500) -> BinanceEndpointResponse:
        return self.__call_api_endpoint(
            BINANCE_KLINES_ENDPOINT,

            {
                "symbol": symbol,
                "interval": interval,
                "startTime": start_time,
                "endTime": end_time,
                "limit": limit
            }
        )


    # Funções especiais para informações específicas

    # Obtém todas as informações sobre um par de moedas

    def get_symbol_info(self, symbol: str) -> BinanceSymbolInfo | None:
        exchange_info = self.__get_exchange_info(symbol)
        average_price = self.__get_average_price(symbol)

        if not (exchange_info.status_code == 200 and average_price.status_code == 200):
            return

        symbols = exchange_info.response["symbols"]
        price = average_price.response["price"]

        if len(symbols) == 0:
            return

        symbol_info = symbols[0]

        binance_symbol_info = BinanceSymbolInfo(
            symbol_info["baseAsset"], symbol_info["quoteAsset"],
            price, symbol_info["status"] == "TRADING",
            symbol_info["orderTypes"], symbol_info["permissionSets"]
        )

        return binance_symbol_info

    # Obtém os candlesticks de um par no período selecionado

    def get_symbol_klines(self, symbol: str, interval: str, start_time: datetime, end_time: datetime, limit: int = 500) -> list[BinanceKline] | None:
        epoch = datetime(1970, 1, 1)

        start = int((start_time - epoch).total_seconds() * 1000)
        end = int((end_time - epoch).total_seconds() * 1000)

        symbol_klines = self.__get_klines(symbol, interval, start, end, limit)

        if symbol_klines.status_code != 200:
            return

        klines = [BinanceKline(k[1], k[2], k[3], k[4]) for k in symbol_klines.response]

        return klines


# Resposta do servidor da Binance

class BinanceEndpointResponse:
    def __init__(self, status_code: int, reason: str = "", response: dict[Any, Any] = {}):
        self.status_code = status_code
        self.reason = reason
        self.response = response


# Tipos especiais por informação

# Informações sobre um símbolo

class BinanceSymbolInfo:
    def __init__(self, base: str, quote: str, price: float, trading_allowed: bool, order_types: list[str | None], permissions: list[str | None]):
        # Informações sobre as moedas que compõe o par: principal (base) e secundária (quote)

        self.base = base
        self.quote = quote

        # Preço da moeda no momento atual
        self.price = price

        # Se é permitido realizar uma operação naquele momento
        self.trading_allowed = trading_allowed

        # Order Types são os tipos de operações disponíveis para o par e Permissions são os níveis de permissão necessária que sua conta precisa ter para determinada operação

        self.order_types = order_types
        self.permissions = permissions

# Representação de candlesticks

class BinanceKline:
    def __init__(self, open: float, high: float, low: float, close: float):
        self.open = ScientificNotation(open)    # Preço de abertura
        self.high = ScientificNotation(high)    # Preço da sombra superior
        self.low = ScientificNotation(low )     # Preço da sombra inferior
        self.close = ScientificNotation(close)  # Preço do fechamento

    # Transforma a representação  do candle em lista de notação normal

    def to_normal_list(self) -> list[float]:
        return [
            self.open.value,
            self.high.value,
            self.low.value,
            self.close.value
        ]

    # Transforma a representação do candle em lista de notação científica

    def to_scientific_notation_list(self) -> list[float]:
        return [
            self.open.mantissa,
            self.open.expoent,
            self.high.mantissa,
            self.high.expoent,
            self.low.mantissa,
            self.low.expoent,
            self.close.mantissa,
            self.close.expoent
        ]
