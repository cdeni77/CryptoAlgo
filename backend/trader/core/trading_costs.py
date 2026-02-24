from typing import Dict

# Shared Coinbase CDE-like fee/contract specs
CONTRACT_SPECS: Dict[str, Dict[str, float | str]] = {
    'BIP': {'units': 0.01,  'min_fee_usd': 0.20, 'fee_pct': 0.0010, 'base': 'BTC'},
    'ETP': {'units': 0.10,  'min_fee_usd': 0.20, 'fee_pct': 0.0010, 'base': 'ETH'},
    'XPP': {'units': 500,   'min_fee_usd': 0.20, 'fee_pct': 0.0010, 'base': 'XRP'},
    'SLP': {'units': 5,     'min_fee_usd': 0.20, 'fee_pct': 0.0010, 'base': 'SOL'},
    'DOP': {'units': 5000,  'min_fee_usd': 0.20, 'fee_pct': 0.0010, 'base': 'DOGE'},
    'BTC': {'units': 0.01,  'min_fee_usd': 0.20, 'fee_pct': 0.0010, 'base': 'BTC'},
    'ETH': {'units': 0.10,  'min_fee_usd': 0.20, 'fee_pct': 0.0010, 'base': 'ETH'},
    'XRP': {'units': 500,   'min_fee_usd': 0.20, 'fee_pct': 0.0010, 'base': 'XRP'},
    'SOL': {'units': 5,     'min_fee_usd': 0.20, 'fee_pct': 0.0010, 'base': 'SOL'},
    'DOGE': {'units': 5000, 'min_fee_usd': 0.20, 'fee_pct': 0.0010, 'base': 'DOGE'},
    'DEFAULT': {'units': 1, 'min_fee_usd': 0.20, 'fee_pct': 0.0010, 'base': 'UNKNOWN'},
}


def get_contract_spec(symbol: str) -> Dict[str, float | str]:
    if symbol in CONTRACT_SPECS:
        return CONTRACT_SPECS[symbol]

    prefix = symbol.split('-')[0] if '-' in symbol else symbol
    if prefix in CONTRACT_SPECS:
        return CONTRACT_SPECS[prefix]

    symbol_upper = symbol.upper()
    for code, spec in CONTRACT_SPECS.items():
        if code == 'DEFAULT':
            continue
        if code in symbol_upper:
            return spec

    return CONTRACT_SPECS['DEFAULT']
