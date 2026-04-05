import asyncio
import json
import logging
import os
import tempfile
import time
from concurrent.futures.thread import ThreadPoolExecutor
from datetime import datetime

from fastapi import UploadFile, File, BackgroundTasks, Form, APIRouter, HTTPException, Depends, Response

from config import Config
from dependencies import get_whisper_service, get_executor, get_config
from services.whisper_service import WhisperService
from utils.memory_utils import cleanup_files
from utils.time_utils import format_time

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/transcribe")
async def transcribe(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    return_timestamps: bool = Form(False),
    whisper_service: WhisperService = Depends(get_whisper_service),
    executor: ThreadPoolExecutor = Depends(get_executor),
    config: Config = Depends(get_config),
):
    try:
        content = await file.read()
        if len(content) > config.api_config.max_file_size:
            raise HTTPException(
                status_code=400,
                detail=f"File size is too big. Maximum {config.api_config.max_file_size // (1024*1024)} MB",
            )

        ext = os.path.splitext(file.filename or "audio")[-1] or ".bin"
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext, dir=config.api_config.temp_dir) as tmp:
            tmp.write(content)
            path_for_transcribe = tmp.name

        logger.info("File: %s (%.1f MB)", file.filename, len(content) / 1024 / 1024)

        start = time.time()

        result = await asyncio.get_running_loop().run_in_executor(
            executor,
            whisper_service.run_transcribe,
            path_for_transcribe,
        )

        elapsed = time.time() - start
        logger.info("Done in %.1f s", elapsed)

        dur = result["segments"][-1]["end"] if result["segments"] else 0.0
        payload: dict = {
            "ok": True,
            "text": result["text"],
            "lang": result["language"],
            "file": file.filename,
            "dur": round(float(dur), 3),
            "conf": round(result["confidence"], 4),
            "sec": round(elapsed, 3),
            "meta": {
                "dev": config.model_config.device.upper(),
                "mdl": config.model_config.model_size,
                "fp16": config.model_config.use_fp16,
            },
            "at": datetime.now().isoformat(),
        }

        if return_timestamps:
            payload["seg"] = [
                [format_time(s["start"]), format_time(s["end"]), s["text"]]
                for s in result["segments"]
            ]

        return Response(
            content=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            media_type="application/json; charset=utf-8",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Transcribe error: %s", e, exc_info=True)
        return Response(
            content=json.dumps({"ok": False, "err": str(e)}, ensure_ascii=False),
            status_code=500,
            media_type="application/json; charset=utf-8",
        )
    finally:
        background_tasks.add_task(cleanup_files, path_for_transcribe)