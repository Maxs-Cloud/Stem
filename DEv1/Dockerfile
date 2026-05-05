FROM python:3.11-slim

WORKDIR /app

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Копируем requirements
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Копируем код
COPY ../Dem /app

# Скачиваем модель предварительно
RUN python -c "from demucs import pretrained; pretrained.get_model('htdemucs')"

# Открываем порт
EXPOSE 8000

# Запускаем API
CMD ["python", "api.py"]