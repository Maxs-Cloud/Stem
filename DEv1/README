# Music Source Separator API

API для разделения музыки на инструментальные дорожки (стемы) с использованием модели Hybrid Transformer Demucs (htdemucs_ft). Проект реализован в рамках тестового задания на позицию ML-инженера.

## Возможности

- Разделение аудио на 4 стема: vocals, drums, bass, other
- REST API на FastAPI с автоматической документацией Swagger
- Скачивание результатов: отдельные WAV-файлы или ZIP-архив
- Оценка качества: метрики SDR, SIR, SAR, ISR на датасете MUSDB18
- Визуализация: спектрограммы и столбчатые диаграммы метрик
- Docker-контейнеризация для простого развёртывания

## Технологический стек

Категория          | Инструменты
-------------------|-------------
ML-модель          | demucs-infer (htdemucs_ft), PyTorch
API                | FastAPI, Uvicorn
Аудио              | Librosa, SoundFile, TorchAudio
Метрики            | museval, MUSDB18
Визуализация       | Matplotlib
Контейнеризация    | Docker, Docker Compose

## Быстрый старт

git clone https://github.com/Maxs-Cloud/music-separator.git
cd music-separator
pip install -r requirements.txt
python main.py

Сервер запустится на http://localhost:8000
Документация Swagger: http://localhost:8000/docs

## Использование

Запуск API сервера:
python main.py --mode api --port 8000

Загрузка аудио и разделение:
curl -X POST "http://localhost:8000/separate" -F "file=@song.mp3"

Ответ:
{
  "status": "success",
  "request_id": "abc123",
  "processing_time_seconds": 45.91,
  "stems": {
    "vocals": "http://localhost:8000/download/abc123/vocals",
    "drums": "http://localhost:8000/download/abc123/drums",
    "bass": "http://localhost:8000/download/abc123/bass",
    "other": "http://localhost:8000/download/abc123/other"
  },
  "zip_archive": "http://localhost:8000/download/abc123/archive"
}

Оценка метрик на MUSDB18:
python main.py --mode metrics --musdb-path ./data/musdb18 --num-tracks 10

Метрики:
- SDR (Source-to-Distortion Ratio) — общее качество разделения
- SIR (Source-to-Interference Ratio) — изоляция от других источников
- SAR (Source-to-Artifacts Ratio) — отсутствие артефактов
- ISR (Image-to-Spatial Ratio) — сохранение пространственных характеристик

## Структура проекта

music-separator/
├── main.py             # Точка входа (API или оценка метрик)
├── api.py              # FastAPI приложение
├── model.py            # Класс DemucsSeparator
├── metrics.py          # Класс MetricsEvaluator
├── requirements.txt    # Python-зависимости
├── Dockerfile          # Docker образ
├── docker-compose.yml  # Docker Compose конфигурация
└── README.txt          # Документация

## Запуск через Docker

docker build -t music-separator .
docker run -p 8000:8000 music-separator

## Запуск через Docker Compose

docker-compose up -d

## Поддерживаемые форматы аудио

WAV (.wav), MP3 (.mp3), FLAC (.flac), OGG (.ogg), M4A (.m4a)
Максимальная длительность аудио: 10 минут.

## Модель

htdemucs_ft (Hybrid Transformer Demucs, fine-tuned)
- SDR 9.0 dB на MUSDB HQ
- Гибридная спектрограммно-волновая архитектура
- Кросс-доменные Transformer-энкодеры

## Аргументы командной строки

python main.py --help

Параметры:
--mode          Режим запуска: api или metrics (по умолчанию: api)
--musdb-path    Путь к датасету MUSDB18 (по умолчанию: ./data/musdb18)
--num-tracks    Количество треков для оценки (по умолчанию: 10)
--model         Модель: htdemucs_ft, htdemucs, mdx_extra (по умолчанию: htdemucs_ft)
--device        Устройство: cuda или cpu (по умолчанию: авто)
--port          Порт для API сервера (по умолчанию: 8000)

## Требования

Python 3.11+
PyTorch 2.0+
Минимум 4 ГБ оперативной памяти (8 ГБ рекомендуется)
Для GPU: CUDA 11.7+

## Контакты

Карпов Максим Дмитриевич
Email: maxkkarp@yandex.ru
Телефон: +7 (950) 0317567
GitHub: https://github.com/Maxs-Cloud

## Лицензия

MIT License
