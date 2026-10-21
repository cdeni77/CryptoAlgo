"""Freeze and version the validated research dataset.

Writes an immutable, hashed parquet plus a manifest recording the eligibility rule,
its justification, and the per-row quality label. Nothing downstream may read the
store directly after this point -- experiments read the frozen file, so a dataset
change is a version bump rather than a silent drift.

Eligibility labels (per RESEARCH_PROTOCOL and D10/D11):

  verified     the quote is from a source whose staleness is bounded and measured
  questionable usable as a FEATURE, not trusted as an executable price
  unsuitable   not usable for execution simulation

The rule is SOURCE-SPECIFIC. `quote_age_seconds` means a Predexon sampling gap for
`backfill` and the sampler's own polling lag for the live recorders, so one global
threshold across both is meaningless (D10).
"""
from __future__ import annotations
import argparse, hashlib, json, logging, os, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s %(message)s')
import harness as H

OUT = HERE / 'datasets'


def label_rows(px: pd.DataFrame, verified_s: float, questionable_s: float) -> pd.Series:
    src = px.quote_source.fillna('unknown')
    age = pd.to_numeric(px.book_age_s, errors='coerce')
    lab = pd.Series('unsuitable', index=px.index, dtype=object)
    is_bf = src.eq('backfill')
    # backfill: the age IS the sampling gap, so it bounds the staleness
    lab[is_bf & age.le(verified_s)] = 'verified'
    lab[is_bf & age.gt(verified_s) & age.le(questionable_s)] = 'questionable'
    # live recorders: the age is the sampler's own lag, already handled by the
    # as-of join on available_time; the book itself is directly observed
    lab[src.isin(('live', 'live_ws'))] = 'verified'
    # live_touch: the decision-instant REST fetch. D5b could not validate the
    # post-2026-09-16 era against any independent source, so it is usable as a
    # feature but not trusted alone as an executable price.
    lab[src.eq('live_touch')] = 'questionable'
    return lab


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--version', default='v2')
    ap.add_argument('--verified-age-s', type=float, default=5.0)
    ap.add_argument('--questionable-age-s', type=float, default=60.0)
    ap.add_argument('--justification', required=True)
    a = ap.parse_args()

    px = H.market_prices(max_book_age_s=1e9)
    lab = H.venue_labels()
    df = px.merge(lab, on=['symbol', 'window_open'], how='inner')
    df['y'] = df.settled_up.astype(int)
    df['quality'] = label_rows(df, a.verified_age_s, a.questionable_age_s)
    df['usable'] = df.quality.eq('verified')
    df['split'] = np.where(df.window_open < H.DEV_END, 'dev', 'val')

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f'market_{a.version}.parquet'
    df.to_parquet(path, index=False)
    h = hashlib.sha256(path.read_bytes()).hexdigest()

    counts = df.quality.value_counts().to_dict()
    by_src = df.groupby(['quote_source', 'quality']).size().unstack(fill_value=0)
    ver = df[df.usable]
    manifest = {
        'version': a.version, 'sha256': h, 'path': str(path.name),
        'rows': int(len(df)),
        'rows_by_quality': {k: int(v) for k, v in counts.items()},
        'verified_rows': int(len(ver)),
        'verified_windows': int(ver.window_open.nunique()),
        'verified_symbol_windows': int(ver.groupby(['symbol', 'window_open']).ngroups),
        'span': [str(df.window_open.min()), str(df.window_open.max())],
        'dev_end': str(H.DEV_END), 'latency_s': H.DECISION_LATENCY_S,
        'eligibility_rule': {
            'backfill_verified_max_age_s': a.verified_age_s,
            'backfill_questionable_max_age_s': a.questionable_age_s,
            'live_live_ws': 'verified (directly observed book; as-of join handles sampler lag)',
            'live_touch': 'questionable (D5b: post-2026-09-16 era unvalidatable)',
        },
        'justification': a.justification,
        'git_commit': os.popen('git rev-parse HEAD').read().strip(),
        'built_at': pd.Timestamp.utcnow().isoformat(),
    }
    (OUT / f'market_{a.version}.manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: v for k, v in manifest.items() if k != 'justification'}, indent=2))
    print('\nquality by source:')
    print(by_src.to_string())
    print(f'\nwrote {path}  sha256={h[:16]}...')


if __name__ == '__main__':
    main()
