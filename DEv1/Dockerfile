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

# Порт
EXPOSE 8000

# Точка входа – запуск API (модель загрузится при старте контейнера)
CMD ["python", "main.py", "--mode", "api"]