@echo off
title Whisper GPU Max Quality API
chcp 65001 > nul
echo ========================================
echo 🚀 ЗАПУСК WHISPER GPU MAX QUALITY API
echo ========================================
echo.
echo Проверка GPU...
python -c "import torch; print('CUDA доступна:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
echo.
echo Запуск сервера...
python main.py
pause