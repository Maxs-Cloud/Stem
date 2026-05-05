from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import tempfile
import shutil
from pathlib import Path
from typing import Dict
import zipfile
import uuid
import time
import os

from model import DemucsSeparator

app = FastAPI(
    title="Music Source Separator API",
    description="API для разделения музыки на инструментальные дорожки",
    version="1.0.0"
)

# Инициализируем модель при запуске
separator = None


@app.on_event("startup")
async def load_model():
    """Загрузка модели при старте сервера."""
    global separator
    separator = DemucsSeparator(model_name='htdemucs')


def validate_audio(filename: str, max_size_mb: int = 50) -> bool:
    """
    Валидация аудиофайла.

    Args:
        filename: имя файла
        max_size_mb: максимальный размер в мегабайтах

    Returns:
        True если файл валидный, иначе вызывает исключение
    """
    # Проверка расширения
    allowed_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    ext = Path(filename).suffix.lower()

    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Неподдерживаемый формат: {ext}. Поддерживаются: {allowed_extensions}"
        )

    return True


def check_audio_duration(file_path: str, max_duration_min: int = 10) -> bool:
    """
    Проверка длительности аудио.

    Args:
        file_path: путь к файлу
        max_duration_min: максимальная длительность в минутах

    Returns:
        True если длительность в пределах лимита
    """
    import librosa

    try:
        duration = librosa.get_duration(filename=file_path)

        if duration > max_duration_min * 60:
            raise HTTPException(
                status_code=400,
                detail=f"Файл слишком длинный: {duration / 60:.1f} мин. Максимум: {max_duration_min} мин."
            )

        return True
    except Exception as e:
        if "HTTPException" in str(type(e)):
            raise
        raise HTTPException(status_code=400, detail=f"Ошибка чтения аудио: {str(e)}")


@app.post("/separate")
async def separate_audio(file: UploadFile = File(...)):
    """
    Разделение аудиофайла на стемы.

    Args:
        file: загружаемый аудиофайл (WAV, MP3, FLAC и др.)

    Returns:
        JSON с ссылками на стемы или ZIP-архив
    """
    # Валидация
    validate_audio(file.filename)

    # Создаем временную директорию
    request_id = str(uuid.uuid4())
    temp_dir = Path(tempfile.gettempdir()) / f"music_sep_{request_id}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Сохраняем загруженный файл
        input_path = temp_dir / file.filename
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Проверяем длительность
        check_audio_duration(str(input_path))

        # Разделяем
        start_time = time.time()
        stems = separator.separate(str(input_path))
        processing_time = time.time() - start_time

        # Опционально: обрезаем тишину
        trimmed_stems = {}
        for name, audio in stems.items():
            trimmed_stems[name] = separator.trim_silence(audio)

        # Сохраняем стемы
        output_dir = temp_dir / "stems"
        stems_paths = {}

        for name, audio in trimmed_stems.items():
            output_path = output_dir / f"{name}.wav"
            output_path.parent.mkdir(exist_ok=True)

            import soundfile as sf
            if audio.ndim == 1:
                sf.write(str(output_path), audio, separator.sample_rate)
            else:
                sf.write(str(output_path), audio.T, separator.sample_rate)

            stems_paths[name] = str(output_path)

        # Создаем ZIP-архив
        zip_path = temp_dir / "stems.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for stem_name, stem_path in stems_paths.items():
                zipf.write(stem_path, f"{Path(file.filename).stem}_{stem_name}.wav")

        return JSONResponse({
            'status': 'success',
            'request_id': request_id,
            'processing_time_seconds': round(processing_time, 2),
            'stems': {
                'vocals': f"/download/{request_id}/vocals",
                'drums': f"/download/{request_id}/drums",
                'bass': f"/download/{request_id}/bass",
                'other': f"/download/{request_id}/other"
            },
            'zip_archive': f"/download/{request_id}/archive",
            'info': {
                'model': separator.model_name,
                'device': separator.device,
                'sample_rate': separator.sample_rate
            }
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")


@app.get("/download/{request_id}/{stem_name}")
async def download_stem(request_id: str, stem_name: str):
    """
    Скачивание отдельного стема.
    """
    temp_dir = Path(tempfile.gettempdir()) / f"music_sep_{request_id}"

    if stem_name == "archive":
        file_path = temp_dir / "stems.zip"
        media_type = "application/zip"
        filename = "stems.zip"
    else:
        file_path = temp_dir / "stems" / f"{stem_name}.wav"
        media_type = "audio/wav"
        filename = f"{stem_name}.wav"

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Результат не найден")

    return FileResponse(
        path=str(file_path),
        media_type=media_type,
        filename=filename
    )


@app.get("/health")
async def health_check():
    """Проверка работоспособности сервиса."""
    return {
        'status': 'healthy',
        'model': separator.model_name,
        'device': separator.device,
        'gpu_available': torch.cuda.is_available()
    }


@app.get("/")
async def root():
    """Корневой эндпоинт с документацией."""
    return {
        'message': 'Music Source Separator API',
        'docs': '/docs',
        'endpoints': {
            'GET /health': 'Проверка состояния',
            'POST /separate': 'Разделение аудио (multipart/form-data)',
            'GET /download/{request_id}/{stem_name}': 'Скачивание результата'
        }
    }


if __name__ == "__main__":
    import uvicorn
    import torch

    uvicorn.run(app, host="0.0.0.0", port=8000)