# Funções gerais genéricas

from market_insights.settings import *

# Verifica se o projeto está em modo de produção
def app_is_prod_mode() -> bool:
    return OPERATION_MODE == "prod"
