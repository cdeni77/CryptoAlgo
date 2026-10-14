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
    """Adopt the artifact's init-score source, refusing the one mode that cannot
    supply it.

    The artifact records which forecaster its correction was fitted on top of,
    and live has no legitimate choice: scoring a market-fitted residual on the
    baseline logit is the silent failure `ForecastModel.verify` exists to catch.
    So the artifact wins.

    A market-init model needs the book to score, and `fetch_quotes` returns {}
    when no Kalshi client was opened (every paper run). Adopting silently would
    turn a loud crash into a NaN prediction every cycle that reads as an
    abstention, so that combination raises.
    """
    # The economic policy the artifact was EVALUATED under. `verify` cannot catch
    # a mismatch here: these change what the strategy DOES, not what the model
    # SAYS. Kelly is also an edge filter, not only a size — `decide()` floors
    # stakes to whole contracts (CLAUDE.md: 0.25 -> 0.10 Kelly moved realised
    # edge per contract +0.99pp -> +3.32pp).
    provenance = dict(getattr(model, 'config_provenance', None) or {})
    # Anything the command line set explicitly wins: adoption is so an artifact
    # trades under the policy it was measured under, not to overrule the
    # operator at the next promotion.
    explicit = set(getattr(config, 'cli_overrides', None) or ())
    economics = {}
    for field in ('kelly_fraction', 'min_edge_pp', 'max_stake_dollars',
                  'max_stake_fraction', 'half_spread_cents', 'compound',
                  # The ensemble gates are part of the policy the artifact was
                  # evaluated under, so they travel with it like the rest.
                  'edge_lower_bound_z', 'require_member_agreement'):
        if field in explicit:
            continue
        value = provenance.get(field)
        # None means the promoting run left it at ITS default, which is the
        # default here too. Adopting a None would erase a real setting.
        if value is not None and value != getattr(config, field, None):
            economics[field] = value
    if economics:
        logger.info('adopting the economics this artifact was measured under: %s',
                    ', '.join(f'{k}={v}' for k, v in sorted(economics.items())))
        config = replace(config, **economics)

    # An artifact that reads the cross-venue group refuses a row without the
    # peer book: `--complete-cases` put the peer on every training row, so a
    # missing one sends LightGBM down NaN branches it never fitted. Explicit wins.
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
    """What identifies the installed artifact on disk, cheaply.

    `promote` installs by atomic rename, so a new artifact is a new inode with
    its own mtime and size; reading the bytes every window is not needed.
    """
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (st.st_ino, st.st_mtime_ns, st.st_size)


def _load_artifact(args, base_config: Config):
    """The installed artifact and the config to trade it under, as `main` builds them."""
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

    Returns `(model, config, signature)`. Loading runs in a thread: a joblib
    load is seconds of blocking work and the stream readers share this loop.
    A refusal — a gate-blocked artifact under `--require-gates`, an init source
    this mode cannot score, an unreadable file — is logged as an ERROR and the
    loop keeps trading what it was already trading.
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
        # Not retried every window: a refused artifact stays refused until
        # the file changes again.
        return model, config, current
    logger.warning('now trading artifact %s (was %s)',
                   getattr(new_model, 'version', '?'),
                   getattr(model, 'version', '?'))
    return new_model, new_config, current


def _installed_version() -> Optional[str]:
    """Which staged artifact is the one on disk, by CONTENT.

    Not "the newest ledger entry": a failed promotion installs nothing, and a
    rollback deliberately writes no ledger entry (a rollback is not a trial).
    `promote` stages each candidate under `models/.staging/<version>/` and
    renames it into place, so the installed file is byte-identical to exactly
    one of them.
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
    """The version recorded on every prediction: the promotion ledger's.

    The staged version whose bytes the file matches; failing that (a file
    promoted before staging existed, or `--model` pointing elsewhere) a content
    hash, so rows from two different artifacts can never share a version.
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
    # The artifact ON DISK, not the newest attempt: the question is "did the
    # model I am about to trade pass?", a property of the file. A failed later
    # promotion or a rollback must not block a passing installed model.
    installed = _installed_version()
    if installed is not None:
        match = frame[frame['version'] == installed]
        if len(match):
            frame = match
    latest = frame.iloc[0]
    # `installed` and `passed` are different questions: `--force` installs a
    # gate-failing artifact with installed=True, so test the gates too.
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
