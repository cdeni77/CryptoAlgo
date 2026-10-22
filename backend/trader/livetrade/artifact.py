"""The promoted artifact: the config it trades under, its version, hot reload, gates."""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import replace
from pathlib import Path
from typing import Optional

from core.config import Config
from core.promotion import LIVE_MODEL, MODELS_ROOT, load_live

logger = logging.getLogger('live')


def config_for_artifact(config, model, *, mode: str):
    """Adopt the artifact's economics and init-score source.

    The artifact's init-score source always wins: a residual fitted on one
    logit is wrong on the other. A market-init model needs a book, so it raises
    outside live mode instead of scoring NaN every cycle.
    """
    # The economic policy the artifact was evaluated under. `verify` cannot
    # catch a mismatch here, because these change what the strategy does.
    # Kelly is also an edge filter: `decide()` floors stakes to whole contracts.
    provenance = dict(getattr(model, 'config_provenance', None) or {})
    # A value set explicitly on the command line wins over the artifact.
    explicit = set(getattr(config, 'cli_overrides', None) or ())
    economics = {}
    for field in ('kelly_fraction', 'min_edge_pp', 'max_stake_dollars',
                  'max_stake_fraction', 'half_spread_cents', 'compound',
                  # The ensemble gates are part of the evaluated policy.
                  'edge_lower_bound_z', 'require_member_agreement',
                  # **The PRICE BASIS the artifact was fitted on**, for the
                  # same reason the init score is adopted: a model fitted on
                  # one basis is wrong on another. `venue` chooses the price
                  # series and `strike_source` the barrier, and the two must
                  # move together -- a BRTI price against a Coinbase OHLC
                  # strike injects a cross-index basis into all ten
                  # strike-derived features.
                  #
                  # Neither is a flag on `scripts.live`, so without this the
                  # trader has NO way to follow a migrated artifact: it would
                  # keep scoring on coinbase_spot for ever while the model was
                  # fitted on cfbenchmarks, and nothing would say so -- the
                  # drift warning fires above 25bp and the basis is about 1bp.
                  'venue', 'strike_source'):
        if field in explicit:
            continue
        value = provenance.get(field)
        # None means the promoting run used the default. Adopting it would
        # erase a real setting.
        if value is not None and value != getattr(config, field, None):
            economics[field] = value
    if economics:
        logger.info('adopting the economics this artifact was measured under: %s',
                    ', '.join(f'{k}={v}' for k, v in sorted(economics.items())))
        config = replace(config, **economics)

    # An artifact that reads cross_venue refuses a row without the peer book.
    # Training rows always carry the peer, so a NaN takes branches never
    # fitted. An explicit flag wins.
    if ('cross_venue' in tuple(getattr(model, 'groups', ()) or ())
            and 'require_peer_book' not in explicit
            and not getattr(config, 'require_peer_book', False)):
        logger.info('this artifact reads cross_venue, so a decision without '
                    'the peer book is refused (--no-require-peer-book overrides)')
        config = replace(config, require_peer_book=True)

    source = getattr(model, 'init_score_source', 'baseline')
    if source == getattr(config, 'init_score_source', 'baseline'):
        return config
    if source == 'market' and mode != 'live':
        raise SystemExit(
            f'this artifact was fitted on the market logit, and {mode} mode '
            f'opens no venue client, so there is no book to read a price from. '
            f'Every window would score NaN and look like an abstention. Run it '
            f'with --mode live (add --dry-run to place nothing).')
    return replace(config, init_score_source=source)


def artifact_signature(path) -> Optional[tuple]:
    """Return a cheap identity for the installed artifact file.

    `promote` installs by atomic rename, so a new artifact has a new inode,
    mtime and size.
    """
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (st.st_ino, st.st_mtime_ns, st.st_size)


def _load_artifact(args, base_config: Config):
    """Load the installed artifact and the config to trade it under."""
    model = (load_live(config=None) if args.model is None
             else __import__('core.model', fromlist=['ForecastModel'])
             .ForecastModel.load(args.model, None))
    if model is None:
        raise RuntimeError('no artifact on disk')
    stamp_version(model, Path(args.model) if args.model
                  else Path(MODELS_ROOT) / LIVE_MODEL)
    config = config_for_artifact(base_config, model, mode=args.mode)
    model.verify(config)
    if not model.deployable:
        raise RuntimeError('the artifact carries no scoring bundle')
    if args.require_gates and not args.force:
        _refuse_if_blocked()
    return model, config


async def reload_if_changed(args, base_config: Config, path, model, config,
                            signature):
    """Swap in a newly installed artifact; on any failure keep the old one.

    Returns `(model, config, signature)`. Loading runs in a thread because the
    stream readers share this event loop. A refused artifact is logged as an
    error and the loop keeps the old model.
    """
    current = artifact_signature(path)
    if current is None or current == signature:
        return model, config, signature
    logger.info('the installed artifact changed on disk; reloading')
    try:
        new_model, new_config = await asyncio.to_thread(
            _load_artifact, args, base_config)
    except (Exception, SystemExit) as exc:                    # noqa: BLE001
        logger.error('could not load the new artifact (%s); still trading %s',
                     str(exc)[:300], getattr(model, 'version', 'the old model'),
                     exc_info=not isinstance(exc, SystemExit))
        # A refused artifact stays refused until the file changes again.
        return model, config, current
    logger.warning('now trading artifact %s (was %s)',
                   getattr(new_model, 'version', '?'),
                   getattr(model, 'version', '?'))
    return new_model, new_config, current


def _installed_version() -> Optional[str]:
    """Return the staged version whose bytes match the installed file.

    The newest ledger entry is not the answer: a failed promotion installs
    nothing, and a rollback writes no ledger entry.
    """
    return _staged_version(Path(MODELS_ROOT) / 'forecast.joblib')


def _staged_version(path) -> Optional[str]:
    import hashlib

    path = Path(path)
    if not path.exists():
        return None
    digest = hashlib.md5(path.read_bytes()).hexdigest()
    staging = Path(MODELS_ROOT) / '.staging'
    if not staging.is_dir():
        return None
    for candidate in sorted(staging.iterdir(), reverse=True):
        staged = candidate / 'forecast.joblib'
        if staged.exists() and hashlib.md5(staged.read_bytes()).hexdigest() == digest:
            return candidate.name
    return None


def artifact_version(path) -> Optional[str]:
    """Return the version recorded on every prediction.

    This is the matching staged version, or else a content hash. Two different
    artifacts never share a version.
    """
    import hashlib

    try:
        staged = _staged_version(path)
        if staged is not None:
            return staged
        return 'md5:' + hashlib.md5(Path(path).read_bytes()).hexdigest()[:12]
    except OSError:
        return None


def stamp_version(model, path) -> None:
    """Put the artifact's version on the model, where `run_cycle` reads it."""
    version = artifact_version(path)
    try:
        model.version = version
    except AttributeError:
        logger.warning('could not record the artifact version on the model')
    logger.info('trading artifact %s', version)


def _refuse_if_blocked() -> None:
    from core.promotion import history

    frame = history()
    if frame.empty:
        raise SystemExit('no promotion attempt recorded; refusing to trade')
    # Judge the artifact on disk, not the newest attempt. A later failed
    # promotion or a rollback must not block a passing installed model.
    installed = _installed_version()
    if installed is not None:
        match = frame[frame['version'] == installed]
        if len(match):
            frame = match
    latest = frame.iloc[0]
    # `--force` installs a gate-failing artifact with installed=True, so test
    # `passed` too.
    if not bool(latest.get('installed')):
        raise SystemExit(
            f"the newest attempt {latest.get('version')} was blocked on "
            f"{latest.get('failed_gates')}. Trading it needs --no-require-gates "
            f"and --force with a written reason.")
    if 'passed' in frame.columns and not bool(latest.get('passed')):
        raise SystemExit(
            f"the newest attempt {latest.get('version')} was force-installed "
            f"with failing gates ({latest.get('failed_gates')}). It is on disk, "
            f"but --require-gates means what it says. Trading it needs "
            f"--no-require-gates and --force with a written reason.")
