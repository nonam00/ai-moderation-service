from dataclasses import dataclass

from pydantic import BaseModel
from typing import List, Optional, Dict, Any

@dataclass(frozen=True)
class Segment(BaseModel):
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class TranscriptionResponse(BaseModel):
    ok: bool
    text: str
    lang: Optional[str]
    file: Optional[str]
    dur: float
    conf: float
    sec: float
    meta: Dict[str, Any]
    at: str
    seg: Optional[List[List[str]]] = None


@dataclass(frozen=True)
class ErrorResponse(BaseModel):
    err: str
    ok: bool = False
