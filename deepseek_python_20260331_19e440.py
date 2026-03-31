import os
import tempfile
import whisper
import torch
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List
import logging
import json
import hashlib
import time
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import subprocess
import shutil
import gc

# ============ НАСТРОЙКА GPU ============
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Определяем устройство и оптимизируем
def setup_gpu():
    """Настройка GPU для максимальной производительности"""
    if torch.cuda.is_available():
        # Получаем информацию о GPU
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        logger.info(f"🎮 GPU обнаружен: {gpu_name}")
        logger.info(f"📊 Количество GPU: {gpu_count}")
        logger.info(f"💾 Видеопамять: {gpu_memory:.1f} GB")
        
        # Оптимизации для GPU
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Выбираем модель в зависимости от памяти
        if gpu_memory >= 8:
            model_size = "large"
            logger.info("✅ Достаточно памяти для модели large")
        elif gpu_memory >= 6:
            model_size = "medium"
            logger.info("✅ Достаточно памяти для модели medium")
        elif gpu_memory >= 4:
            model_size = "small"
            logger.info("✅ Достаточно памяти для модели small")
        else:
            model_size = "base"
            logger.warning("⚠️ Мало видеопамяти, используется модель base")
        
        device = "cuda"
        use_fp16 = True  # Используем FP16 для скорости
        
    elif torch.backends.mps.is_available():
        # Для Mac M1/M2
        device = "mps"
        model_size = "medium"
        use_fp16 = False
        logger.info("🍎 Используется Apple MPS (Metal Performance Shaders)")
        
    else:
        device = "cpu"
        model_size = "base"
        use_fp16 = False
        logger.info("💻 GPU не найден, используется CPU")
    
    return device, model_size, use_fp16

# Настраиваем GPU
DEVICE, MODEL_SIZE, USE_FP16 = setup_gpu()

# ============ КОНФИГУРАЦИЯ ============
class Config:
    MODEL_SIZE = MODEL_SIZE
    DEVICE = DEVICE
    USE_FP16 = USE_FP16
    MAX_FILE_SIZE = 500 * 1024 * 1024  # 500 МБ
    TEMP_DIR = tempfile.mkdtemp()
    CACHE_ENABLED = True
    CACHE_DIR = os.path.join(TEMP_DIR, "cache")
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # Параметры для GPU
    BATCH_SIZE = 4 if DEVICE == "cuda" else 1
    NUM_WORKERS = 2 if DEVICE == "cuda" else 1
    
    # Языки с высоким качеством
    HIGH_QUALITY_LANGUAGES = ["ru", "en", "es", "fr", "de", "it", "pt", "nl", "pl", "uk", "zh", "ja", "ko"]
    
    # Оптимальные настройки для разных языков
    LANGUAGE_PROMPTS = {
        "ru": "Это русская речь. Обратите особое внимание на окончания слов, падежи и грамматику.",
        "en": "This is English speech. Pay attention to pronunciation and context.",
        "es": "Esta es una conversación en español. Presta atención a la pronunciación y el contexto.",
        "fr": "Il s'agit d'un discours en français. Faites attention à la prononciation.",
        "de": "Dies ist eine deutsche Rede. Achten Sie auf die Aussprache.",
        "zh": "这是中文语音。注意发音和语调。",
        "ja": "これは日本語の音声です。発音に注意してください。",
        "ko": "이것은 한국어 음성입니다. 발음에 주의하세요.",
    }

config = Config()

# ============ ЗАГРУЗКА МОДЕЛИ С ОПТИМИЗАЦИЕЙ ДЛЯ GPU ============
print("\n" + "="*70)
print("🚀 ЗАПУСК WHISPER GPU MAX QUALITY")
print("="*70)
print(f"📊 Модель: {config.MODEL_SIZE.upper()}")
print(f"💻 Устройство: {config.DEVICE.upper()}")
if config.DEVICE == "cuda":
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 Видеопамять: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"🚀 FP16: {'Включено' if config.USE_FP16 else 'Выключено'}")
print("="*70 + "\n")

print("⏳ Загрузка модели Whisper...")

# Загружаем модель с оптимизациями
model = whisper.load_model(config.MODEL_SIZE, device=config.DEVICE)

# Оптимизации для GPU
if config.DEVICE == "cuda":
    if config.USE_FP16:
        model = model.half()  # Используем FP16 для скорости
        logger.info("✅ FP16 активирован")
    
    # Оптимизация для инференса
    model.eval()
    torch.set_grad_enabled(False)
    
    # Оптимизация памяти
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    logger.info("✅ Оптимизация GPU завершена")

print("✅ Модель загружена успешно!")
print("\n" + "="*70 + "\n")

# ============ FASTAPI APP ============
app = FastAPI(
    title="Whisper GPU Max Quality API",
    description="Максимально качественное распознавание речи с использованием GPU",
    version="4.0.0"
)

# CORS для веб-приложений
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Пул потоков для обработки (оптимизирован для GPU)
executor = ThreadPoolExecutor(max_workers=config.NUM_WORKERS)

# ============ ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ============
def get_file_hash(file_path: str) -> str:
    """Вычисление хеша файла для кэширования"""
    with open(file_path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()

def preprocess_audio(file_path: str) -> str:
    """Предварительная обработка аудио для улучшения качества"""
    try:
        output_path = file_path.replace('.', '_processed.')
        
        # Оптимальные параметры для Whisper
        cmd = [
            'ffmpeg', '-i', file_path,
            '-af', 'loudnorm=I=-16:LRA=11:TP=-1.5,highpass=f=200,lowpass=f=3000',
            '-ac', '1',  # Моно
            '-ar', '16000',  # 16kHz оптимально для Whisper
            '-y',  # Перезаписывать
            output_path
        ]
        
        subprocess.run(cmd, capture_output=True, check=True)
        return output_path
    except Exception as e:
        logger.warning(f"Предобработка не удалась: {e}")
        return file_path

def transcribe_with_advanced_settings(file_path: str, language: Optional[str], task: str) -> dict:
    """Транскрипция с максимальными настройками качества и GPU оптимизациями"""
    
    # Определяем язык для оптимальных настроек
    if language and language != "auto":
        lang_code = language
    else:
        # Быстрое определение языка с использованием GPU
        try:
            audio = whisper.load_audio(file_path)
            audio = whisper.pad_or_trim(audio)
            
            # Перемещаем на GPU если доступно
            if config.DEVICE == "cuda":
                audio_tensor = torch.from_numpy(audio).float()
                audio_tensor = audio_tensor.to(config.DEVICE)
                mel = whisper.log_mel_spectrogram(audio_tensor)
            else:
                mel = whisper.log_mel_spectrogram(audio)
            
            _, probs = model.detect_language(mel)
            lang_code = max(probs, key=probs.get)
            logger.info(f"Определен язык: {lang_code} (вероятность: {probs[lang_code]:.2f})")
        except Exception as e:
            logger.warning(f"Ошибка определения языка: {e}")
            lang_code = None
    
    # Оптимальные настройки для разных языков
    best_settings = {
        "temperature": 0.0,
        "compression_ratio_threshold": 2.4,
        "logprob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "condition_on_previous_text": True,
        "verbose": False,
        "word_timestamps": True,
        "fp16": config.USE_FP16  # Используем FP16 для скорости на GPU
    }
    
    # Добавляем язык
    if lang_code:
        best_settings["language"] = lang_code
        best_settings["initial_prompt"] = config.LANGUAGE_PROMPTS.get(lang_code, "")
    
    # Добавляем задачу
    best_settings["task"] = task
    
    # Для русского языка дополнительные улучшения
    if lang_code == "ru":
        best_settings["initial_prompt"] = (
            "Это русская речь. Внимательно слушайте окончания слов, "
            "правильно определяйте падежи и грамматические конструкции. "
            "Записывайте текст грамотно, с правильными окончаниями."
        )
        best_settings["compression_ratio_threshold"] = 2.2
    
    # Выполняем транскрипцию
    result = model.transcribe(file_path, **best_settings)
    
    # Пост-обработка текста
    text = result["text"].strip()
    
    return {
        "text": text,
        "language": result.get("language", lang_code),
        "segments": result.get("segments", []),
        "confidence": get_confidence_score(result.get("segments", []))
    }

def get_confidence_score(segments: List) -> float:
    """Вычисление уверенности распознавания"""
    if not segments:
        return 0.0
    
    confidences = []
    for segment in segments:
        if "avg_logprob" in segment:
            confidences.append(np.exp(segment["avg_logprob"]))
    
    return np.mean(confidences) if confidences else 0.0

def format_time(seconds: float) -> str:
    """Форматирование времени"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"

def cleanup_files(*file_paths):
    """Удаление временных файлов и очистка GPU памяти"""
    for file_path in file_paths:
        if file_path and os.path.exists(file_path):
            try:
                os.unlink(file_path)
            except:
                pass
    
    # Очистка GPU памяти после обработки
    if config.DEVICE == "cuda":
        torch.cuda.empty_cache()
        gc.collect()

# ============ API ENDPOINTS ============

@app.get("/")
async def root():
    """Информация о сервисе"""
    gpu_info = {}
    if config.DEVICE == "cuda":
        gpu_info = {
            "name": torch.cuda.get_device_name(0),
            "memory_total": f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB",
            "memory_used": f"{torch.cuda.memory_allocated(0) / 1e9:.1f} GB",
            "memory_free": f"{(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1e9:.1f} GB"
        }
    
    return {
        "service": "Whisper GPU Max Quality API",
        "version": "4.0.0",
        "model": config.MODEL_SIZE,
        "device": config.DEVICE.upper(),
        "fp16": config.USE_FP16,
        "gpu_info": gpu_info,
        "status": "ready",
        "features": [
            "GPU ускорение (CUDA)",
            "Максимальное качество распознавания",
            "Автоопределение языка",
            "Шумоподавление",
            "Временные метки слов",
            "Кэширование результатов",
            "Поддержка 100+ языков",
            "FP16 оптимизация"
        ]
    }

@app.get("/gpu-info")
async def gpu_info():
    """Информация о GPU"""
    if config.DEVICE != "cuda":
        return {"status": "GPU не доступен", "device": config.DEVICE}
    
    return {
        "status": "GPU доступен",
        "device_name": torch.cuda.get_device_name(0),
        "memory_total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
        "memory_allocated_gb": torch.cuda.memory_allocated(0) / 1e9,
        "memory_cached_gb": torch.cuda.memory_reserved(0) / 1e9,
        "temperature": torch.cuda.temperature(0) if hasattr(torch.cuda, 'temperature') else "N/A",
        "utilization": torch.cuda.utilization(0) if hasattr(torch.cuda, 'utilization') else "N/A"
    }

@app.get("/languages")
async def get_languages():
    """Список поддерживаемых языков с настройками качества"""
    return {
        "languages": {
            "auto": {"name": "Автоопределение", "quality": "Максимальное"},
            "ru": {"name": "Русский", "quality": "Оптимизировано", "features": ["грамматика", "падежи"]},
            "en": {"name": "English", "quality": "Native", "features": ["accents", "dialects"]},
            "es": {"name": "Español", "quality": "Alta calidad", "features": ["acentos"]},
            "fr": {"name": "Français", "quality": "Haute qualité", "features": ["prononciation"]},
            "de": {"name": "Deutsch", "quality": "Hohe Qualität", "features": ["Aussprache"]},
            "it": {"name": "Italiano", "quality": "Alta qualità", "features": ["pronuncia"]},
            "pt": {"name": "Português", "quality": "Alta qualidade", "features": ["sotaques"]},
            "zh": {"name": "中文", "quality": "高质量", "features": ["声调"]},
            "ja": {"name": "日本語", "quality": "高品質", "features": ["アクセント"]},
            "ko": {"name": "한국어", "quality": "고품질", "features": ["발음"]}
        },
        "total": 100,
        "gpu_accelerated": config.DEVICE == "cuda",
        "note": "Поддерживаются все языки из модели Whisper"
    }

@app.post("/transcribe")
async def transcribe(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Аудиофайл (любой формат)"),
    language: Optional[str] = Form(None, description="Код языка (ru, en, es...) или auto"),
    task: str = Form("transcribe", description="transcribe/translate"),
    return_timestamps: bool = Form(False, description="Вернуть временные метки"),
    return_segments: bool = Form(False, description="Вернуть сегменты"),
    use_cache: bool = Form(True, description="Использовать кэш")
):
    """
    Максимально качественное распознавание речи с использованием GPU
    
    - Использует GPU для ускорения (CUDA)
    - Поддерживаются все аудиоформаты
    - Автоматическое шумоподавление
    - Оптимизация для каждого языка
    - Высокая точность даже при сложных условиях
    - Временные метки слов
    - Кэширование результатов
    """
    
    temp_file = None
    processed_file = None
    
    try:
        # Сохраняем файл
        ext = file.filename.split('.')[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}", dir=config.TEMP_DIR) as tmp:
            content = await file.read()
            tmp.write(content)
            temp_file = tmp.name
        
        # Проверка размера
        file_size_mb = len(content) / (1024 * 1024)
        if file_size_mb > config.MAX_FILE_SIZE / (1024 * 1024):
            raise HTTPException(status_code=400, detail=f"Файл слишком большой. Максимум {config.MAX_FILE_SIZE / (1024 * 1024):.0f} МБ")
        
        # Кэширование
        file_hash = get_file_hash(temp_file)
        cache_key = f"{file_hash}_{language}_{task}"
        cache_file = os.path.join(config.CACHE_DIR, f"{cache_key}.json")
        
        if use_cache and config.CACHE_ENABLED and os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_result = json.load(f)
            logger.info(f"Результат из кэша для {file.filename}")
            return JSONResponse(content=cached_result)
        
        # Предобработка аудио
        logger.info(f"Обработка файла: {file.filename} ({file_size_mb:.1f} МБ)")
        processed_file = preprocess_audio(temp_file)
        
        # Распознавание
        logger.info(f"Начало распознавания... Язык: {language or 'auto'}")
        logger.info(f"Устройство: {config.DEVICE.upper()} | FP16: {config.USE_FP16}")
        start_time = time.time()
        
        result = await asyncio.get_event_loop().run_in_executor(
            executor,
            transcribe_with_advanced_settings,
            processed_file,
            language,
            task
        )
        
        elapsed_time = time.time() - start_time
        
        # Логируем использование GPU
        if config.DEVICE == "cuda":
            gpu_memory = torch.cuda.memory_allocated(0) / 1e9
            logger.info(f"Использовано GPU памяти: {gpu_memory:.2f} GB")
        
        logger.info(f"Распознавание завершено за {elapsed_time:.1f} сек")
        
        # Формирование ответа
        response = {
            "success": True,
            "text": result["text"],
            "language": result["language"],
            "language_name": get_language_name(result["language"]),
            "task": task,
            "filename": file.filename,
            "duration": result["segments"][-1]["end"] if result["segments"] else 0,
            "confidence": result["confidence"],
            "processing_time": elapsed_time,
            "device_used": config.DEVICE.upper(),
            "model_used": config.MODEL_SIZE,
            "fp16_used": config.USE_FP16,
            "timestamp": datetime.now().isoformat()
        }
        
        if return_timestamps:
            response["timestamps"] = [
                {
                    "start": format_time(seg["start"]),
                    "end": format_time(seg["end"]),
                    "text": seg["text"]
                }
                for seg in result["segments"]
            ]
        
        if return_segments:
            response["segments"] = result["segments"]
        
        # Сохраняем в кэш
        if use_cache and config.CACHE_ENABLED:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(response, f, ensure_ascii=False, indent=2)
        
        return JSONResponse(content=response)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка: {e}", exc_info=True)
        return JSONResponse(
            content={"success": False, "error": str(e)},
            status_code=500
        )
    
    finally:
        # Очистка
        background_tasks.add_task(cleanup_files, temp_file, processed_file)

@app.get("/health")
async def health():
    """Проверка здоровья сервиса с информацией о GPU"""
    gpu_info = {}
    if config.DEVICE == "cuda":
        gpu_info = {
            "available": True,
            "name": torch.cuda.get_device_name(0),
            "memory_total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
            "memory_used_gb": torch.cuda.memory_allocated(0) / 1e9,
            "memory_free_gb": (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1e9
        }
    
    return {
        "status": "healthy",
        "model": config.MODEL_SIZE,
        "device": config.DEVICE.upper(),
        "fp16": config.USE_FP16,
        "gpu": gpu_info,
        "cache_size": len(os.listdir(config.CACHE_DIR)) if os.path.exists(config.CACHE_DIR) else 0,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None
    }

@app.post("/clear-gpu-memory")
async def clear_gpu_memory():
    """Принудительная очистка GPU памяти"""
    if config.DEVICE == "cuda":
        torch.cuda.empty_cache()
        gc.collect()
        return {
            "success": True,
            "message": "GPU память очищена",
            "memory_freed_gb": torch.cuda.memory_allocated(0) / 1e9
        }
    return {"success": False, "message": "GPU не доступен"}

@app.delete("/cache")
async def clear_cache():
    """Очистка кэша"""
    try:
        if os.path.exists(config.CACHE_DIR):
            shutil.rmtree(config.CACHE_DIR)
            os.makedirs(config.CACHE_DIR)
        return {"success": True, "message": "Кэш очищен"}
    except Exception as e:
        return {"success": False, "error": str(e)}

# ============ ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ============

def get_language_name(lang_code: str) -> str:
    """Получение названия языка"""
    languages = {
        "ru": "Русский",
        "en": "English",
        "es": "Español",
        "fr": "Français",
        "de": "Deutsch",
        "it": "Italiano",
        "pt": "Português",
        "nl": "Nederlands",
        "pl": "Polski",
        "uk": "Українська",
        "zh": "中文",
        "ja": "日本語",
        "ko": "한국어",
        "ar": "العربية",
        "hi": "हिन्दी"
    }
    return languages.get(lang_code, lang_code)

# ============ ЗАПУСК ============
if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*70)
    print("✨ WHISPER GPU MAX QUALITY API - ГОТОВ К РАБОТЕ! ✨")
    print("="*70)
    print(f"🎯 Модель: {config.MODEL_SIZE.upper()}")
    print(f"💻 Устройство: {config.DEVICE.upper()}")
    if config.DEVICE == "cuda":
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 Видеопамять: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"🚀 FP16: Включен (ускорение ~2x)")
    print(f"📁 Кэш: {config.CACHE_DIR}")
    print(f"🌍 Языки: 100+ с автоопределением")
    print("="*70)
    print("\n📖 Документация: http://localhost:8000/docs")
    print("🎮 GPU статус: http://localhost:8000/gpu-info")
    print("🌐 Список языков: http://localhost:8000/languages")
    print("🏥 Проверка: http://localhost:8000/health")
    print("\n💡 Рекомендации для максимальной производительности:")
    print("  • Убедитесь, что драйверы NVIDIA обновлены")
    print("  • Для максимальной скорости используйте CUDA 11.8+")
    print("  • Модель large требует 8+ GB видеопамяти")
    print("  • Используйте FP16 для ускорения в 2 раза")
    print("\n⚡ Нажмите CTRL+C для остановки сервера\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        timeout_keep_alive=300
    )