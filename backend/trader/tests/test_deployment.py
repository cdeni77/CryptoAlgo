"""What the container is actually told to run, checked against what it accepts.

Three deployment defects shipped at once, and all three were invisible to a test
suite that only ever ran scripts directly:

- `docker-compose.yml` passed `--debug` to `scripts.live_orchestrator`, which
  declares no such flag. argparse exits 2 before any work, and with
  `restart: unless-stopped` the trader sat in a permanent crash loop. The
  healthcheck only opened a Postgres connection, so it reported healthy the whole
  time.
- `duckdb` is imported at module scope by `core/datastore.py`, which every script
  reaches through `scripts/_common.py`, and it was absent from
  `requirements.txt`. The image could not import a single script. It worked in
  development only because the dev environment happened to have it.
- A `./configs:/app/configs:ro` mount pointed at a repo-root directory that does
  not exist, so Docker created an empty one and mounted it *over* the real
  `configs/` that the bind mount had just brought in. `find_cost_config()` then
  found nothing and every contract was priced at the hardcoded 10bp/side — the
  exact bug the comment above the mount claimed was fixed.

The shape of all three is the same: the deployed configuration disagreed with the
code, and nothing compared them.
"""

from __future__ import annotations

import argparse
import importlib
import re
from pathlib import Path

import pytest

yaml = pytest.importorskip('yaml')

REPO_ROOT = Path(__file__).resolve().parents[3]
TRADER_ROOT = Path(__file__).resolve().parents[1]
COMPOSE = REPO_ROOT / 'docker-compose.yml'


@pytest.fixture(scope='module')
def compose() -> dict:
    if not COMPOSE.exists():
        pytest.skip(f'no compose file at {COMPOSE}')
    return yaml.safe_load(COMPOSE.read_text())


def _trader_services(compose: dict) -> dict[str, dict]:
    """Services whose command runs a module out of this package."""
    return {
        name: service
        for name, service in compose.get('services', {}).items()
        if '-m scripts.' in str(service.get('command', ''))
    }


def _module_and_flags(command: str) -> tuple[str, list[str]]:
    tokens = str(command).split()
    module = tokens[tokens.index('-m') + 1]
    return module, [t for t in tokens if t.startswith('--')]


# ---------------------------------------------------------------------------
# The command the container runs
# ---------------------------------------------------------------------------


def test_every_compose_command_names_a_module_that_exists(compose):
    for name, service in _trader_services(compose).items():
        module, _ = _module_and_flags(service['command'])
        importlib.import_module(module)  # raises if it does not exist


def test_every_compose_flag_is_declared_by_its_script(compose):
    """The `--debug` regression.

    An undeclared flag is not a warning — argparse exits 2 before the script
    does anything, which under `restart: unless-stopped` is a crash loop that the
    healthcheck cannot see.
    """
    problems: list[str] = []

    for name, service in _trader_services(compose).items():
        module_name, flags = _module_and_flags(service['command'])
        module = importlib.import_module(module_name)

        # Read the flags the script declares, without running main(): several
        # scripts build their parser inline rather than exposing `parse_args`.
        declared: set[str] = set()
        source = Path(module.__file__).read_text()
        declared.update(re.findall(r"add_argument\(\s*'(--[a-z0-9-]+)'", source))
        # Scripts that share the common data surface inherit its flags.
        if 'add_data_arguments' in source:
            common = (TRADER_ROOT / 'scripts' / '_common.py').read_text()
            declared.update(re.findall(r"add_argument\(\s*'(--[a-z0-9-]+)'", common))

        for flag in flags:
            if flag not in declared:
                problems.append(f'{name}: {module_name} does not accept {flag}')

    assert not problems, '\n'.join(problems)


def test_the_compose_command_actually_parses(compose):
    """Stronger than the flag check: run the real parser over the real argv."""
    for name, service in _trader_services(compose).items():
        module_name, _ = _module_and_flags(service['command'])
        module = importlib.import_module(module_name)
        if not hasattr(module, 'parse_args'):
            continue

        tokens = str(service['command']).split()
        argv = tokens[tokens.index('-m') + 2:]

        import sys
        saved = sys.argv
        sys.argv = [module_name, *argv]
        try:
            module.parse_args()
        except SystemExit as exc:
            pytest.fail(f'{name}: `{module_name} {" ".join(argv)}` exits {exc.code}')
        finally:
            sys.argv = saved


# ---------------------------------------------------------------------------
# What the image contains
# ---------------------------------------------------------------------------


def test_every_module_scope_third_party_import_is_in_requirements():
    """`duckdb` was imported by the store and named nowhere in requirements.

    Checked against the import statements rather than a hand-maintained list,
    because the hand-maintained list is what was wrong.
    """
    requirements = (TRADER_ROOT / 'requirements.txt').read_text().lower()

    # Distribution name where it differs from the import name.
    distributions = {
        'sklearn': 'scikit-learn',
        'yaml': 'pyyaml',
        'dateutil': 'python-dateutil',
        'psycopg2': 'psycopg2-binary',
        'lightgbm': 'lightgbm',
    }
    stdlib_or_local = {
        'core', 'scripts', 'data_collection', 'tests', 'config', 'configs',
        '__future__', 'annotations',
    }

    missing: set[str] = set()
    for path in [*(TRADER_ROOT / 'core').glob('*.py'),
                 *(TRADER_ROOT / 'scripts').glob('*.py'),
                 *(TRADER_ROOT / 'data_collection').glob('*.py')]:
        for line in path.read_text().splitlines():
            match = re.match(r'^(?:import|from)\s+([a-zA-Z_][a-zA-Z0-9_]*)', line)
            if not match:
                continue
            top = match.group(1)
            if top in stdlib_or_local:
                continue
            try:  # standard library needs no requirement
                spec = importlib.util.find_spec(top)
            except (ImportError, ValueError):
                spec = None
            if spec is None or spec.origin in (None, 'built-in', 'frozen'):
                continue
            if 'site-packages' not in str(spec.origin) and 'dist-packages' not in str(spec.origin):
                continue
            name = distributions.get(top, top)
            if name.lower() not in requirements:
                missing.add(f'{name} (imported as {top}, e.g. {path.name})')

    assert not missing, (
        'imported at module scope but absent from requirements.txt, so the image '
        'cannot import them:\n  ' + '\n  '.join(sorted(missing))
    )


def test_the_research_store_dependencies_are_pinned():
    """The specific regression, named, so it cannot come back quietly."""
    requirements = (TRADER_ROOT / 'requirements.txt').read_text().lower()

    assert 'duckdb' in requirements
    assert 'pyarrow' in requirements


# ---------------------------------------------------------------------------
# What the mounts do to it
# ---------------------------------------------------------------------------


def test_no_mount_shadows_the_cost_config(compose):
    """A bind mount over `configs/` is how every contract got mispriced.

    `configs/` sits inside `backend/trader` precisely so the Docker build context
    includes it. Mounting anything at `/app/configs` throws that away, and a host
    path that does not exist is worse than one that does: Docker creates it empty.
    """
    for name, service in compose.get('services', {}).items():
        for volume in service.get('volumes', []) or []:
            if not isinstance(volume, str) or ':' not in volume:
                continue
            source, target = volume.split(':')[:2]
            if target.rstrip('/').endswith('/configs'):
                pytest.fail(
                    f'{name} mounts {source} at {target}, masking the fee schedule '
                    f'that the build context already provides'
                )
            if source.startswith('./') and not (REPO_ROOT / source[2:]).exists():
                pytest.fail(
                    f'{name} mounts {source}, which does not exist — Docker will '
                    f'create it empty and mount it over {target}'
                )


def test_the_paper_engine_can_reach_the_store_and_the_fee_schedule(compose):
    """Its two documented failure modes are configuration, not code.

    `FundingSource` reads the research store and `_build_config` loads the venue
    schedule. With no data volume and neither env var, funding was never accrued
    and every fill was priced at the hardcoded default — the two bugs the module
    docstring says it exists to fix.
    """
    service = compose['services'].get('paper-engine')
    if service is None:
        pytest.skip('no paper-engine service')

    environment = {
        entry.split('=')[0]: entry.split('=', 1)[1]
        for entry in service.get('environment', []) or []
        if '=' in entry
    }
    assert 'RESEARCH_STORE' in environment, 'funding cannot be accrued without the store'
    assert 'COST_CONFIG' in environment, 'fills would be priced at the hardcoded default'

    data_mounts = [
        v for v in service.get('volumes', []) or []
        if isinstance(v, str) and v.split(':')[1:2] == ['/app/data']
    ]
    assert data_mounts, 'no /app/data volume, so RESEARCH_STORE points at nothing'


def test_the_paper_engine_reads_the_same_store_the_trader_writes(compose):
    """One store, or the engine accrues funding the orchestrator never collected."""
    services = compose['services']
    trader = {e.split('=')[0]: e.split('=', 1)[1]
              for e in services['trader'].get('environment', []) if '=' in e}
    engine = {e.split('=')[0]: e.split('=', 1)[1]
              for e in services['paper-engine'].get('environment', []) if '=' in e}

    assert trader.get('RESEARCH_STORE') == engine.get('RESEARCH_STORE')

    def data_volume(name):
        return {v.split(':')[0] for v in services[name].get('volumes', [])
                if isinstance(v, str) and v.split(':')[1:2] == ['/app/data']}

    assert data_volume('trader') == data_volume('paper-engine'), (
        'different /app/data sources: the engine would read an empty store'
    )


# ---------------------------------------------------------------------------
# The environment the container runs with
# ---------------------------------------------------------------------------

# Variables the runtime reads, not this codebase: the base image, the driver,
# the Postgres entrypoint, or a Vite build. Each is a deliberate exemption.
_RUNTIME_ENV = {
    'TZ',                       # glibc / the base image
    'PYTHONUNBUFFERED',         # CPython
    'PYTHONPATH',               # CPython
    'HTTPS_PROXY', 'HTTP_PROXY', 'NO_PROXY',
    'POSTGRES_USER', 'POSTGRES_PASSWORD', 'POSTGRES_DB',   # postgres entrypoint
    'PGDATA',
    'NODE_ENV',                 # vite / node
    'CHOKIDAR_USEPOLLING',      # vite file watcher in a container
}


def _declared_env(service: dict) -> set[str]:
    """Variable names a compose service sets, from either env syntax."""
    env = service.get('environment') or {}
    if isinstance(env, dict):
        return {str(k) for k in env}
    names = set()
    for entry in env:
        text = str(entry)
        # Skip a bare `- NAME` pass-through, which sets nothing here.
        if '=' in text:
            names.add(text.split('=', 1)[0].strip())
    return names


def test_every_environment_variable_compose_sets_is_read_by_something(compose):
    """A knob nothing reads is worse than a missing one.

    `LEVERAGE=4` sat in the trader service, documented in AGENTS.md with a
    default, and read by no code at all — so an operator lowering it watched the
    book keep sizing at 4x. It multiplies target notional in
    `execution.size_from_forecast` and divides margin, so the silence was
    expensive. It is wired now, and this test is what stops the next one.

    The check is deliberately loose about *where* the read happens (trader, API,
    or the frontend's `import.meta.env`) — the failure mode is a variable read
    nowhere, not one read in the wrong package.
    """
    sources: list[str] = []
    for root in (REPO_ROOT / 'backend', REPO_ROOT / 'frontend' / 'src'):
        if not root.exists():
            continue
        for path in root.rglob('*'):
            if path.suffix in {'.py', '.ts', '.tsx'} and '__pycache__' not in path.parts:
                sources.append(path.read_text(errors='ignore'))
    haystack = '\n'.join(sources)
    assert haystack, 'found no source to search, so this test proves nothing'

    unread: dict[str, list[str]] = {}
    for name, service in compose.get('services', {}).items():
        for variable in sorted(_declared_env(service)):
            if variable in _RUNTIME_ENV:
                continue
            # VITE_* reach the browser through import.meta.env, not os.getenv.
            if variable not in haystack:
                unread.setdefault(variable, []).append(name)

    assert not unread, (
        'compose sets variables no code reads:\n  '
        + '\n  '.join(f'{k} (in {", ".join(v)})' for k, v in sorted(unread.items()))
        + '\n\nEither wire it or delete it. A knob that silently does nothing is '
          'how LEVERAGE=4 stayed authoritative while an operator lowered it.'
    )


def test_the_trader_and_paper_engine_agree_on_leverage(compose):
    """Two services size the same book. They cannot disagree about leverage.

    `scripts.signals` sizes a position with `config.leverage`; the paper engine
    then reserves margin as `notional / leverage` and caps contracts at
    `cash * leverage`. If the two read different values the engine either
    reserves the wrong margin or clips the signal writer's own size — and the
    engine used to read no value at all, taking the `Config` default of 4
    whatever the deployment set.
    """
    services = compose.get('services', {})
    values = {}
    for name in ('trader', 'paper-engine'):
        service = services.get(name)
        if service is None:
            pytest.skip(f'no {name} service in compose')
        declared = {
            entry.split('=', 1)[0].strip(): entry.split('=', 1)[1].strip()
            for entry in (service.get('environment') or [])
            if isinstance(entry, str) and '=' in entry
        }
        assert 'LEVERAGE' in declared, (
            f'{name} does not set LEVERAGE, so it falls back to the Config '
            f'default regardless of the deployment'
        )
        values[name] = declared['LEVERAGE']

    assert len(set(values.values())) == 1, (
        f'the services disagree about leverage: {values}'
    )


def test_the_orchestrator_forwards_the_horizon_and_leverage_to_every_step():
    """Both have to be identical across the feature build, training and signals.

    The horizon sets the purge width between train and test, is recorded in the
    model's provenance, and drives `effective_observations`. It was unreachable
    from `live_orchestrator` entirely — no flag, no env var — which pinned every
    containerised run to the profile default of 96h. On the 398 days of CDE
    history that exist, 96h carries about 70 effective observations against the
    ~200 the gates need, so the loop could not train a promotable model at all.

    Leverage moved into the same shared set for the same reason: it sizes the
    position in `scripts.signals` and bounds the risk budget in both, so a step
    that saw a different value would size against a different book.
    """
    import sys as _sys

    import scripts.live_orchestrator as orchestrator

    original = _sys.argv
    try:
        _sys.argv = ['live_orchestrator', '--horizon', '24', '--leverage', '2']
        args = orchestrator.parse_args()
    finally:
        _sys.argv = original

    forwarded = orchestrator._data_arguments(args)
    assert '--horizon' in forwarded and forwarded[forwarded.index('--horizon') + 1] == '24'
    assert '--leverage' in forwarded and forwarded[forwarded.index('--leverage') + 1] == '2.0'

    # Unset must stay unset rather than becoming a number: the profile hold is
    # the documented fallback, and 0 is not a valid horizon.
    try:
        _sys.argv = ['live_orchestrator']
        bare = orchestrator.parse_args()
    finally:
        _sys.argv = original
    assert bare.horizon is None
    assert '--horizon' not in orchestrator._data_arguments(bare)
