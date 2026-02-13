from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class OpsActionResponse(BaseModel):
    action: str
    status: str
    detail: str
    pid: Optional[int] = None


class OpsLogEntry(BaseModel):
    raw: str
    timestamp: Optional[datetime] = None
    level: Optional[str] = None
    message: Optional[str] = None


class OpsLogsResponse(BaseModel):
    entries: List[OpsLogEntry]


class OpsStatusResponse(BaseModel):
    pipeline_running: bool
    training_running: bool
    phase: str
    symbol: Optional[str] = None
    metrics: Dict[str, float] = Field(default_factory=dict)
    last_run_time: Optional[datetime] = None
    next_run_time: Optional[datetime] = None
    log_file: str
