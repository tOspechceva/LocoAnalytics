#!/bin/bash

echo "=========================================="
echo "   Locomotive Analytics - Безопасный Запуск   "
echo "=========================================="

# 1. Detect if we handle non-ASCII path issues
WORK_DIR="/tmp/LocoAnalytics_Exec"
SOURCE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "[1/4] Подготовка безопасного окружения..."
# echo "Источник: $SOURCE_DIR"
# echo "Цель: $WORK_DIR"

# Create target directory
mkdir -p "$WORK_DIR"

# Copy files (excluding source venv to avoid copying broken entironment)
# We use rsync if available for speed, else cp
if command -v rsync &> /dev/null; then
    # Sync everything except venv and this script itself
    rsync -a --exclude 'venv' --exclude 'fix_and_start.command' "$SOURCE_DIR/" "$WORK_DIR/"
else
    # Fallback to CP
    cp -R "$SOURCE_DIR/"* "$WORK_DIR/" 2>/dev/null
    # If cp copied the venv folder, remove it from target to be safe
    rm -rf "$WORK_DIR/venv"
fi

# Move to the safe ASCII path
cd "$WORK_DIR"

# 2. Clean up potential bad artifacts from previous runs
# Remove .pth files from USER library if they exist (just to be safe)
TARGET_SITE="$HOME/Library/Python/3.9/lib/python/site-packages"
if [ -d "$TARGET_SITE" ]; then
    rm -f "$TARGET_SITE"/*.pth 2>/dev/null
fi

# 3. Setup Venv
echo "[2/4] Настройка Python окружения (это может занять время)..."
# Check if a valid python is inside venv, if not recreate
if [ ! -f "venv/bin/python3" ]; then
    echo "Создание виртуального окружения..."
    rm -rf venv
    python3 -I -m venv venv
fi

# 4. Install Dependencies
echo "[3/4] Установка библиотек..."
source venv/bin/activate
pip install -r requirements.txt --quiet

# 5. Run App
echo "[4/4] Запуск приложения..."
echo "Открываю браузер..."
streamlit run app.py

# Keep terminal open on error
if [ $? -ne 0 ]; then
    echo "КРИТИЧЕСКАЯ ОШИБКА: Приложение не запустилось."
    echo "Пожалуйста, проверьте подключение к интернету для установки библиотек."
    read -n 1
fi
