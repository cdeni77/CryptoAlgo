import os
import re
import subprocess
import threading
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Deque, Dict, List, Optional

from models.ops import OpsLogEntry, OpsStatusResponse


_LOG_LINE_PATTERN = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:,\d+)?)\s*\|\s*(?P<level>[A-Z]+)\s*\|\s*.*?\|\s*(?P<msg>.*)$"
)
_METRIC_PATTERN = re.compile(r"\b(AUC|OOS_SHARPE|TRADE_COUNT)\s*[=:]\s*(-?\d+(?:\.\d+)?)\b", re.IGNORECASE)
_SYMBOL_PATTERN = re.compile(r"\b(BTC|ETH|SOL|XRP|DOGE)(?:[-_/]PERP)?\b", re.IGNORECASE)


class OpsService:
    def __init__(self):
        self._lock = threading.Lock()
        self._pipeline_proc: Optional[subprocess.Popen] = None
        self._training_proc: Optional[subprocess.Popen] = None
        self._parallel_proc: Optional[subprocess.Popen] = None
        self._phase: str = "idle"
        self._symbol: Optional[str] = None
        self._metrics: Dict[str, float] = {}
        self._last_run_time: Optional[datetime] = None
        self._next_run_time: Optional[datetime] = None

        self._trader_dir = self._resolve_trader_dir()
        self._logs_dir = self._trader_dir / "logs"
        self._logs_dir.mkdir(parents=True, exist_ok=True)
        self._log_file = Path(os.environ.get("TRADER_LOG_FILE", self._logs_dir / "trader.log"))

    def _resolve_trader_dir(self) -> Path:
        env_dir = os.environ.get("TRADER_DIR")
        if env_dir:
            p = Path(env_dir)
            if p.exists():
                return p

        here = Path(__file__).resolve()
        candidates = [
            here.parents[3] / "backend" / "trader",  # repo checkout
            here.parents[2] / "trader",  # legacy layout
            Path("/trader"),  # docker-compose mount target
            Path("/app/trader"),
        ]
        for candidate in candidates:
            if (candidate / "live_orchestrator.py").exists():
                return candidate

        raise RuntimeError("Could not locate trader directory. Set TRADER_DIR environment variable.")

    @property
    def log_file(self) -> Path:
        return self._log_file

    def _spawn(self, cmd: List[str], extra_env: Optional[Dict[str, str]] = None) -> subprocess.Popen:
        log_handle = self._log_file.open("a", encoding="utf-8")
        env = os.environ.copy()
        # Ensure Python workers flush stdout/stderr continuously so the UI log tail
        # reflects progress in near-real time.
        env.setdefault("PYTHONUNBUFFERED", "1")
        if extra_env:
            env.update({k: str(v) for k, v in extra_env.items() if v is not None})

        proc = subprocess.Popen(
            cmd,
            cwd=self._trader_dir,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        self._last_run_time = datetime.utcnow()
        return proc

    def start_pipeline(self) -> int:
        with self._lock:
            if self._pipeline_proc and self._pipeline_proc.poll() is None:
                return self._pipeline_proc.pid

            self._phase = "pipeline_running"
            self._next_run_time = datetime.utcnow() + timedelta(hours=1)
            self._pipeline_proc = self._spawn(["python", "run_pipeline.py", "--skip-backfill"])
            return self._pipeline_proc.pid

    def stop_pipeline(self) -> bool:
        with self._lock:
            if not self._pipeline_proc or self._pipeline_proc.poll() is not None:
                self._phase = "idle"
                return False

            self._pipeline_proc.terminate()
            self._pipeline_proc.wait(timeout=10)
            self._phase = "idle"
            self._next_run_time = None
            return True

    def retrain(self, train_window_days: int = 90, retrain_every_days: int = 7, debug: bool = False) -> int:
        with self._lock:
            if self._training_proc and self._training_proc.poll() is None:
                return self._training_proc.pid

            self._phase = "training"
            self._next_run_time = datetime.utcnow() + timedelta(hours=1)
            cmd = [
                "python",
                "live_orchestrator.py",
                "--retrain-only",
                "--run-once",
                "--train-window-days",
                str(train_window_days),
                "--retrain-every-days",
                str(retrain_every_days),
            ]
            if debug:
                cmd.append("--debug")
            self._training_proc = self._spawn(cmd)
            return self._training_proc.pid

    def launch_parallel(
        self,
        trials: int = 200,
        jobs: int = 16,
        coins: str = "BTC,ETH,SOL,XRP,DOGE",
        plateau_patience: int = 80,
        plateau_min_delta: float = 0.02,
        plateau_warmup: int = 40,
    ) -> int:
        with self._lock:
            if self._parallel_proc and self._parallel_proc.poll() is None:
                return self._parallel_proc.pid

            self._phase = "optimization"
            self._parallel_proc = self._spawn(
                [
                    "python",
                    "parallel_launch.py",
                    "--trials",
                    str(trials),
                    "--jobs",
                    str(jobs),
                    "--coins",
                    coins,
                    "--plateau-patience",
                    str(plateau_patience),
                    "--plateau-min-delta",
                    str(plateau_min_delta),
                    "--plateau-warmup",
                    str(plateau_warmup),
                ]
            )
            return self._parallel_proc.pid

    def train_from_scratch(
        self,
        backfill_days: int = 30,
        include_oi: bool = True,
        debug: bool = False,
        threshold: float = 0.74,
        min_auc: float = 0.54,
        leverage: int = 4,
        exclude_symbols: str = "BIP,DOP",
    ) -> int:
        with self._lock:
            if self._training_proc and self._training_proc.poll() is None:
                return self._training_proc.pid

            self._phase = "training"
            self._next_run_time = datetime.utcnow() + timedelta(hours=1)
            cmd = [
                "python",
                "live_orchestrator.py",
                "--run-once",
                "--backfill-days",
                str(backfill_days),
            ]
            if include_oi:
                cmd.append("--include-oi")
            if debug:
                cmd.append("--debug")

            env = {
                "SIGNAL_THRESHOLD": str(threshold),
                "MIN_AUC": str(min_auc),
                "LEVERAGE": str(leverage),
                "EXCLUDE_SYMBOLS": exclude_symbols,
            }
            self._training_proc = self._spawn(cmd, extra_env=env)
            return self._training_proc.pid

    def _parse_timestamp(self, value: str) -> Optional[datetime]:
        for fmt in ("%Y-%m-%d %H:%M:%S,%f", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
        return None

    def _classify_phase(self, text: str) -> Optional[str]:
        lower = text.lower()
        if "backfill" in lower:
            return "backfill"
        if "funding" in lower:
            return "funding"
        if "new candle" in lower or "ticker" in lower:
            return "live_trading"
        if "launching" in lower and "workers" in lower:
            return "optimization"
        if "train" in lower or "auc" in lower:
            return "training"
        return None

    def _tail_lines(self, limit: int) -> List[str]:
        if not self._log_file.exists():
            return []

        buf: Deque[str] = deque(maxlen=limit)
        with self._log_file.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                stripped = line.rstrip("\n")
                if stripped:
                    buf.append(stripped)
        return list(buf)

    def _to_log_entry(self, raw: str) -> OpsLogEntry:
        match = _LOG_LINE_PATTERN.match(raw)
        if not match:
            return OpsLogEntry(raw=raw)

        ts = self._parse_timestamp(match.group("ts"))
        msg = match.group("msg")
        return OpsLogEntry(raw=raw, timestamp=ts, level=match.group("level"), message=msg)

    def _refresh_from_logs(self, lines: List[str]) -> None:
        for line in lines:
            ts_match = _LOG_LINE_PATTERN.match(line)
            if ts_match:
                parsed_ts = self._parse_timestamp(ts_match.group("ts"))
                if parsed_ts:
                    self._last_run_time = parsed_ts

            phase = self._classify_phase(line)
            if phase:
                self._phase = phase

            symbol_match = _SYMBOL_PATTERN.search(line)
            if symbol_match:
                self._symbol = symbol_match.group(1).upper()

            for metric, value in _METRIC_PATTERN.findall(line):
                key = metric.upper()
                self._metrics[key] = float(value)

    def get_logs(self, limit: int = 200) -> List[OpsLogEntry]:
        lines = self._tail_lines(limit)
        self._refresh_from_logs(lines)
        return [self._to_log_entry(line) for line in lines]

    def get_status(self) -> OpsStatusResponse:
        lines = self._tail_lines(200)
        self._refresh_from_logs(lines)

        pipeline_running = bool(self._pipeline_proc and self._pipeline_proc.poll() is None)
        training_running = bool(self._training_proc and self._training_proc.poll() is None)
        parallel_running = bool(self._parallel_proc and self._parallel_proc.poll() is None)

        if not pipeline_running and not training_running and not parallel_running and self._phase in {"pipeline_running", "optimization"}:
            self._phase = "idle"

        return OpsStatusResponse(
            pipeline_running=pipeline_running,
            training_running=training_running,
            parallel_running=parallel_running,
            phase=self._phase,
            symbol=self._symbol,
            metrics=self._metrics,
            last_run_time=self._last_run_time,
            next_run_time=self._next_run_time,
            log_file=str(self._log_file),
        )


ops_service = OpsService()
