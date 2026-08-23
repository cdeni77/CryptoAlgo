"""The one route that starts a process.

This container holds the Coinbase credentials, so a route that runs a script is
the highest-value target in the deployment. Two independent controls guard it,
and either alone leaves a hole:

* `require_token` gates it on a shared secret and **fails closed** — with no
  `API_TOKEN` configured it returns 503 rather than running unauthenticated. A
  default-open deployment is the failure mode worth designing against, because
  nobody notices it until someone else does.
* `validate_job_args` restricts what may be passed, because authentication says
  *who* may launch, not *what*. An authenticated caller still should not be able
  to hand a research script an arbitrary filesystem path.

`JOBS` is an allow-list of module names, not a pattern. Discovering scripts from
the filesystem would mean any file that appears under `scripts/` becomes
remotely runnable.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from security import require_token, validate_job_args

logger = logging.getLogger('api.jobs')
router = APIRouter(prefix='/jobs', tags=['jobs'])

TRADER_DIR = Path(os.getenv('TRADER_DIR', '/app'))

# Module name -> what it does. Only these can be launched, and the list is
# deliberately short: the long-running research scripts, not the live loop.
# Starting the orchestrator or the paper engine from a web request would mean two
# copies racing over one account.
JOBS: dict[str, str] = {
    'scripts.scrape': 'fetch one-minute Coinbase spot bars into SQLite',
    'scripts.sync_store': 'copy SQLite rows into the Parquet research store',
    'scripts.baseline': 'fit and report the barrier baseline',
    'scripts.evaluate': 'walk-forward evaluation with gates and cost stress',
    'scripts.train': 'fit one model for inspection',
    'scripts.promote': 'evaluate a candidate and install it, gates permitting',
}


class LaunchRequest(BaseModel):
    args: Optional[list[str]] = None


@router.get('')
def list_jobs():
    return {'jobs': [{'module': name, 'description': what}
                     for name, what in sorted(JOBS.items())]}


@router.post('/{job:path}', dependencies=[Depends(require_token)])
def launch(job: str, body: LaunchRequest | None = None):
    if job not in JOBS:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            detail=f'unknown job {job!r}; allowed: {sorted(JOBS)}',
        )
    args = validate_job_args(body.args if body else None)
    command = [sys.executable, '-m', job, *args]
    logger.info('launching %s', ' '.join(command))
    process = subprocess.Popen(  # noqa: S603 - module from an allow-list, args validated
        command, cwd=str(TRADER_DIR),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return {'job': job, 'pid': process.pid, 'args': args,
            'note': 'output goes to the container log, not to this response'}
