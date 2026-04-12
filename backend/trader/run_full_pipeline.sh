#!/bin/bash
# Full pipeline: compute_features -> prune_features -> parallel_launch
# Run from: backend/trader/
set -e

cd "$(dirname "$0")"

# Load env
export $(grep -v '^#' ../../.env | grep -v '^$' | grep -v 'COINBASE_API_SECRET' | xargs 2>/dev/null) || true
eval $(python -c "
from dotenv import dotenv_values
env = dotenv_values('../../.env')
for k,v in env.items():
    v2 = v.replace(\"'\", \"'\\\"'\\\"'\")
    print(f\"export {k}='{v2}'\")
" 2>/dev/null) || true

echo "=============================================="
echo "STEP 1: COMPUTE FEATURES (all 9 coins)"
echo "=============================================="
python -c "
import sys; sys.path.insert(0, '.')
from dotenv import load_dotenv; load_dotenv('../../.env')
import scripts.compute_features as cf
cf.main()
" 2>&1 | tee /tmp/compute_features.log

echo ""
echo "=============================================="
echo "STEP 2: PRUNE FEATURES (SHAP, top-30 per coin)"
echo "=============================================="
python -c "
import sys; sys.path.insert(0, '.')
from dotenv import load_dotenv; load_dotenv('../../.env')
import sys
sys.argv = ['prune_features.py', '--top-n', '30', '--n-splits', '5', '--val-fraction', '0.20']
import scripts.prune_features as pf
pf.main()
" 2>&1 | tee /tmp/prune_features.log

echo ""
echo "=============================================="
echo "STEP 3: PARALLEL OPTIMIZATION (robust_annual)"
echo "=============================================="
python -c "
import sys; sys.path.insert(0, '.')
from dotenv import load_dotenv; load_dotenv('../../.env')
import sys, asyncio
sys.argv = [
    'parallel_launch.py',
    '--coins', 'BTC,ETH,SOL,XRP,DOGE,AVAX,ADA,LINK,LTC',
    '--preset', 'robust_annual',
    '--trials', '400',
    '--workers', 'auto',
    '--jobs', '1',
    '--parallel-mode', 'coin-seed',
    '--sampler-seeds', '42,1337,7',
]
import scripts.parallel_launch as pl
asyncio.run(pl.main())
" 2>&1 | tee /tmp/parallel_launch.log

echo ""
echo "=============================================="
echo "DONE. Check /tmp/parallel_launch.log for results."
echo "=============================================="
