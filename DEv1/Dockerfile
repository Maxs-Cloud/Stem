FROM python:3.11-slim

WORKDIR /app

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    ffmpeg \
        libsndfile1 \
            && rm -rf /var/lib/apt/lists/*

            # Копируем и устанавливаем Python-зависимости
            COPY requirements.txt .
            RUN pip install --no-cache-dir -r requirements.txt

            # Копируем ВСЕ файлы проекта
            COPY . /app

            # Предварительно скачиваем модель demucs-infer (НЕ оригинальный demucs!)
            RUN python -c "from demucs_infer.pretrained import get_model; get_model('htdemucs_ft')"

            # Открываем порт
            EXPOSE 8000

            # Запускаем API
            CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]