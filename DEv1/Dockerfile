FROM python:3.11-slim

WORKDIR /app

# Системные зависимости
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Python-зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Код проекта
COPY . /app

# Скачать модель при сборке (требуется стабильный интернет)
RUN python -c "from demucs_infer.pretrained import get_model; get_model('htdemucs_ft')"

# Порт
EXPOSE 8000

# Точка входа – запуск API (модель загрузится при старте контейнера)
CMD ["python", "main.py", "--mode", "api"]