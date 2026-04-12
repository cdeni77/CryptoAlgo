from typing import Dict

# Coinbase CDE contract specs — user-verified sizes, 0.1% taker fee, no per-contract minimum
CONTRACT_SPECS: Dict[str, Dict[str, float | str]] = {
    'BIP': {'units': 0.01,   'min_fee_usd': 0.0, 'fee_pct': 0.0010, 'base': 'BTC'},
    'ETP': {'units': 0.10,   'min_fee_usd': 0.0, 'fee_pct': 0.0010, 'base': 'ETH'},
    'XPP': {'units': 500,    'min_fee_usd': 0.0, 'fee_pct': 0.0010, 'base': 'XRP'},
    'SLP': {'units': 5,      'min_fee_usd': 0.0, 'fee_pct': 0.0010, 'base': 'SOL'},
    'DOP': {'units': 5000,   'min_fee_usd': 0.0, 'fee_pct': 0.0010, 'base': 'DOGE'},
    'AVP': {'units': 10,     'min_fee_usd': 0.0, 'fee_pct': 0.0010, 'base': 'AVAX'},  # 10 units
    'AVAX': {'units': 10,    'min_fee_usd': 0.0, 'fee_pct': 0.0010, 'base': 'AVAX'},
    'ADP': {'units': 1000,   'min_fee_usd': 0.0, 'fee_pct': 0.0010, 'base': 'ADA'},
    'ADA': {'units': 1000,   'min_fee_usd': 0.0, 'fee_pct': 0.0010, 'base': 'ADA'},
    'LNP': {'units': 50,     'min_fee_usd': 0.0, 'fee_pct': 0.0010, 'base': 'LINK'},  # 50 units
    'LINK': {'units': 50,    'min_fee_usd': 0.0, 'fee_pct': 0.0010, 'base': 'LINK'},
    'LCP': {'units': 5,      'min_fee_usd': 0.0, 'fee_pct': 0.0010, 'base': 'LTC'},   # 5 units
    'LTC': {'units': 5,      'min_fee_usd': 0.0, 'fee_pct': 0.0010, 'base': 'LTC'},
    # ── Batch 3 — new 20DEC30-CDE additions 2026-04-03 ──────────────────────
    'NER':  {'units': 500,      'min_fee_usd': 0.0, 'fee_pct': 0.0010, 'base': 'NEAR'},
    'NEAR': {'units': 500,      'min_fee_usd': 0.0, 'fee_pct': 0.0010, 'base': 'NEAR'},
    'SUP':  {'units': 500,      'min_fee_usd': 0.0, 'fee_pct': 0.0010, 'base': 'SUI'},
    'SUI':  {'units': 500,      'min_fee_usd': 0.0, 'fee_pct': 0.0010, 'base': 'SUI'},
    'BCP':  {'units': 1,        'min_fee_usd': 0.0, 'fee_pct': 0.0010, 'base': 'BCH'},
    'BCH':  {'units': 1,        'min_fee_usd': 0.0, 'fee_pct': 0.0010, 'base': 'BCH'},
    'XLP':  {'units': 5000,     'min_fee_usd': 0.0, 'fee_pct': 0.0010, 'base': 'XLM'},
    'XLM':  {'units': 5000,     'min_fee_usd': 0.0, 'fee_pct': 0.0010, 'base': 'XLM'},
    'POP':  {'units': 100,      'min_fee_usd': 0.0, 'fee_pct': 0.0010, 'base': 'DOT'},
    'DOT':  {'units': 100,      'min_fee_usd': 0.0, 'fee_pct': 0.0010, 'base': 'DOT'},
    'SHP':  {'units': 10000,    'min_fee_usd': 0.0, 'fee_pct': 0.0010, 'base': '1000SHIB'},
    'SHIB': {'units': 10000,    'min_fee_usd': 0.0, 'fee_pct': 0.0010, 'base': '1000SHIB'},
    'PEP':  {'units': 100000,   'min_fee_usd': 0.0, 'fee_pct': 0.0010, 'base': '1000PEPE'},
    'PEPE': {'units': 100000,   'min_fee_usd': 0.0, 'fee_pct': 0.0010, 'base': '1000PEPE'},
    # Ticker aliases for legacy symbol resolution
    'BTC': {'units': 0.01,   'min_fee_usd': 0.0, 'fee_pct': 0.0010, 'base': 'BTC'},
    'ETH': {'units': 0.10,   'min_fee_usd': 0.0, 'fee_pct': 0.0010, 'base': 'ETH'},
    'XRP': {'units': 500,    'min_fee_usd': 0.0, 'fee_pct': 0.0010, 'base': 'XRP'},
    'SOL': {'units': 5,      'min_fee_usd': 0.0, 'fee_pct': 0.0010, 'base': 'SOL'},
    'DOGE': {'units': 5000,  'min_fee_usd': 0.0, 'fee_pct': 0.0010, 'base': 'DOGE'},
    'DEFAULT': {'units': 1,  'min_fee_usd': 0.0, 'fee_pct': 0.0010, 'base': 'UNKNOWN'},
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
