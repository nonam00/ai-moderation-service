@echo off
title Установка Whisper GPU
echo ========================================
echo 🚀 УСТАНОВКА WHISPER С ПОДДЕРЖКОЙ GPU
echo ========================================
echo.

echo Проверка наличия CUDA...
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

echo.
echo Удаление старой версии PyTorch...
pip uninstall torch torchaudio -y

echo.
echo Установка PyTorch с CUDA 11.8...
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

echo.
echo Установка остальных зависимостей...
pip install fastapi uvicorn[standard] openai-whisper numpy ffmpeg-python python-multipart aiofiles python-dotenv

echo.
echo ========================================
echo ✅ УСТАНОВКА ЗАВЕРШЕНА!
echo ========================================
echo.
echo Запустите run.bat для старта сервера
pause