"""The cross-origin policy.

The CORS list used to end with `"*"`, which made every other entry decorative:
any page the browser had open could call the API with credentials.
"""

from __future__ import annotations

from security import allowed_origins


def test_read_only_routes_stay_open(client):
    """The dashboard polls these. Gating them buys nothing the origin policy does not."""
    assert client.get('/').status_code == 200
    assert client.get('/model/').status_code == 200


def test_wildcard_origin_is_filtered_out(monkeypatch):
    """`*` made every other entry in the list decorative."""
    monkeypatch.setenv('CORS_ALLOW_ORIGINS', 'http://localhost:3000,*')

    origins = allowed_origins()

    assert '*' not in origins
    assert origins == ['http://localhost:3000']


def test_an_all_wildcard_list_falls_back_to_localhost(monkeypatch):
    monkeypatch.setenv('CORS_ALLOW_ORIGINS', '*')

    assert allowed_origins() == ['http://localhost:3000']


def test_unset_origins_default_to_local_development(monkeypatch):
    monkeypatch.delenv('CORS_ALLOW_ORIGINS', raising=False)

    origins = allowed_origins()

    assert '*' not in origins
    assert all(o.startswith('http://localhost') or o.startswith('http://127.0.0.1')
               for o in origins)
