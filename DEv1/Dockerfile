FROM python:3.11-slim

WORKDIR /app

# Системные зависимости для аудио-библиотек
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Python-зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Код проекта
COPY . /app

# Прогрев модели при сборке, чтобы не ждать при старте контейнера
RUN python -c "from demucs_infer.pretrained import get_model; get_model('htdemucs_ft')"

EXPOSE 8000

# Запуск API через main.py (режим api)
CMD ["python", "main.py", "--mode", "api"]