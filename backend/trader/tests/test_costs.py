"""The fee model against the venue's own order tickets.

Everything this repo concludes about tradeability is a comparison between a
forecast and a round trip, so the round trip is not a modelling choice — it is a
measurement, and it has to be pinned to observations rather than to a schedule
document. Coinbase publishes a member commission table (group A $0.75/contract,
group B $0.10) that its retail app does not charge; the app's tickets are what
this account actually pays.
"""

from __future__ import annotations

import json

import pytest

from core.config import Config, find_cost_config
from core.costs import (
    get_contract_spec,
    load_exchange_cost_assumptions,
    per_contract_fee,
)
from core.execution import entry_cost
from core.targets import round_trip_cost


@pytest.fixture(scope='module')
def schedule_path():
    path = find_cost_config()
    if path is None:
        pytest.skip('no cost config on the search path')
    return path


def test_the_schedule_reproduces_the_venue_s_own_order_tickets(schedule_path):
    """Every recorded ticket, to the cent.

    The tickets span 3.2x in notional, which is what makes them decisive. Two
    observations at similar size — $782 of BIP and $740 of XPP — cannot separate
    a flat percentage from a flat dollar amount, and for most of a day they did
    not: 0.115% and 0.116% look like one rate. The $242 ETP ticket implies
    0.149%, so no single percentage fits all three, and `0.10% + $0.12/contract`
    fits every one to under half a cent.

    Tolerance is a cent because the app quotes to the cent. That is tight enough
    to reject each model this replaced: `max()` of the two components gives
    $0.78 / $0.74 / $0.24, and the old $0.75-on-BIP-and-ETP table gives $1.53 /
    $0.86 / $0.99.
    """
    assumptions = load_exchange_cost_assumptions(schedule_path)
    config = Config().with_cost_assumptions(schedule_path)

    tickets = assumptions.observed_app_fees
    assert len(tickets) >= 3, 'the fee model is fitted to these; do not drop them'

    notionals = [t['notional_usd'] for t in tickets]
    assert max(notionals) / min(notionals) > 2.0, (
        'the tickets no longer span enough notional to separate a percentage '
        'fee from a per-contract one'
    )

    for ticket in tickets:
        symbol, price = ticket['contract'], float(ticket['underlying_price'])
        contracts = int(ticket.get('contracts', 1))

        notional = get_contract_spec(symbol).notional(contracts, price)
        assert notional == pytest.approx(ticket['notional_usd'], abs=0.05), (
            f'{symbol}: contract units disagree with the recorded notional'
        )

        modelled = entry_cost(contracts, price, symbol, config)
        assert modelled == pytest.approx(ticket['app_fee_usd'], abs=0.01), (
            f'{symbol}: the model charges ${modelled:.4f} where the venue '
            f'charged ${ticket["app_fee_usd"]:.2f} on ${notional:.2f} of notional'
        )


def test_no_single_percentage_fits_the_tickets(schedule_path):
    """Why the per-contract term has to be there at all.

    If one flat percentage could explain the tickets, the commission would be an
    unnecessary parameter and the honest model would be the simpler one. It
    cannot: fitting the rate to the largest ticket underprices the smallest by
    more than the cent the app quotes to.
    """
    tickets = load_exchange_cost_assumptions(schedule_path).observed_app_fees
    biggest = max(tickets, key=lambda t: t['notional_usd'])
    smallest = min(tickets, key=lambda t: t['notional_usd'])

    rate = biggest['app_fee_usd'] / biggest['notional_usd']
    implied = rate * smallest['notional_usd']

    assert abs(implied - smallest['app_fee_usd']) > 0.01, (
        'a flat percentage now explains every ticket, which would make the '
        'per-contract commission an unneeded parameter'
    )


def test_the_commission_makes_small_contracts_dearer_per_dollar(schedule_path):
    """The cost ordering across the book, and it is not the one it used to be.

    A fixed number of dollars is a larger share of a smaller notional, so the
    contracts with the least notional each are the expensive ones per dollar
    traded. Under the old per-symbol table ETP was dearest because of an
    explicit $0.75, and BIP nearly so; now ETP is dearest only because 0.1 ETH
    is ~$242 while 0.01 BTC is ~$782, which is a fact about contract sizes
    rather than about the fee.
    """
    config = Config().with_cost_assumptions(schedule_path)

    # Same rate everywhere.
    rates = {s: per_contract_fee(s, config) for s in ('BIP', 'ETP', 'XPP', 'DOP')}
    assert len(set(rates.values())) == 1, rates

    prices = {'BIP': 78_000.0, 'ETP': 2_425.0, 'XPP': 1.48, 'DOP': 0.20}
    costs = {s: round_trip_cost(s, p, config) for s, p in prices.items()}
    notionals = {s: get_contract_spec(s).notional(1, p) for s, p in prices.items()}

    by_cost = sorted(costs, key=costs.get, reverse=True)
    by_notional = sorted(notionals, key=notionals.get)
    assert by_cost == by_notional, (
        f'cost order {by_cost} should be the inverse of notional order '
        f'{sorted(notionals, key=notionals.get, reverse=True)}'
    )


def test_the_round_trip_is_the_same_order_of_magnitude_across_the_book(schedule_path):
    """A sanity bound on the number every conclusion here divides by.

    With one percentage and one commission the round trip is ~23bp on a
    thousand-dollar contract and ~30bp on a $242 one. The old table produced a
    6bp-to-65bp spread, an order of magnitude, which is what made "XPP and SLP
    are the only affordable contracts" look like a finding. If a schedule change
    reopens that spread, the conclusions that rest on it need revisiting, so
    fail here rather than let it pass quietly.
    """
    config = Config().with_cost_assumptions(schedule_path)
    prices = {'BIP': 78_000.0, 'ETP': 2_425.0, 'XPP': 1.48, 'DOP': 0.20,
              'SLP': 150.0, 'ADP': 0.55, 'LNP': 15.0, 'LCP': 85.0}

    costs = {s: round_trip_cost(s, p, config) * 10_000 for s, p in prices.items()}
    assert min(costs.values()) > 15.0, costs
    assert max(costs.values()) < 45.0, costs
    assert max(costs.values()) / min(costs.values()) < 2.5, costs
