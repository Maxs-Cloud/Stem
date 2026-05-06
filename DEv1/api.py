from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import shutil
from pathlib import Path
from typing import Dict
import zipfile
import uuid
import time
import torch
import soundfile as sf
import librosa
import warnings
warnings.filterwarnings('ignore')

# Импорт вашего класса из отдельного файла
from model import DemucsSeparator

# ==================== FastAPI App ====================
app = FastAPI(
    title="Music Source Separator API",
    description="API для разделения музыки на инструментальные дорожки",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

separator = None

@app.on_event("startup")
async def load_model():
    global separator
    separator = DemucsSeparator(model_name='htdemucs_ft')

def validate_audio(filename: str) -> bool:
    allowed_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    ext = Path(filename).suffix.lower()
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Неподдерживаемый формат: {ext}"
        )
    return True

def check_audio_duration(file_path: str, max_duration_min: int = 10) -> bool:
    try:
        duration = librosa.get_duration(filename=file_path)
        if duration > max_duration_min * 60:
            raise HTTPException(
                status_code=400,
                detail=f"Файл слишком длинный: {duration/60:.1f} мин"
            )
        return True
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка: {str(e)}")

@app.post("/separate")
async def separate_audio(file: UploadFile = File(...), request: Request = None):
    validate_audio(file.filename)

    request_id = str(uuid.uuid4())
    temp_dir = Path(tempfile.gettempdir()) / f"music_sep_{request_id}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        input_path = temp_dir / file.filename
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        check_audio_duration(str(input_path))

        start_time = time.time()
        stems = separator.separate(str(input_path))
        processing_time = time.time() - start_time

        output_dir = temp_dir / "stems"
        stems_paths = {}
        for name, audio in stems.items():
            output_path = output_dir / f"{name}.wav"
            output_path.parent.mkdir(exist_ok=True)
            sf.write(str(output_path), audio.T, separator.sample_rate)
            stems_paths[name] = str(output_path)

        zip_path = temp_dir / "stems.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for stem_name, stem_path in stems_paths.items():
                zipf.write(stem_path, f"{Path(file.filename).stem}_{stem_name}.wav")

        base_url = str(request.base_url).rstrip("/") if request else "http://localhost:8000"

        return JSONResponse({
            'status': 'success',
            'request_id': request_id,
            'processing_time_seconds': round(processing_time, 2),
            'stems': {
                'vocals': f"{base_url}/download/{request_id}/vocals",
                'drums': f"{base_url}/download/{request_id}/drums",
                'bass': f"{base_url}/download/{request_id}/bass",
                'other': f"{base_url}/download/{request_id}/other"
            },
            'zip_archive': f"{base_url}/download/{request_id}/archive",
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

    return FileResponse(path=str(file_path), media_type=media_type, filename=filename)

@app.get("/health")
async def health_check():
    if separator is None:
        return {'status': 'model not loaded'}
    return {
        'status': 'healthy',
        'model': separator.model_name,
        'device': separator.device,
        'gpu_available': torch.cuda.is_available()
    }

@app.get("/")
async def root():
    return {
        'message': 'Music Source Separator API',
        'docs': '/docs',
        'endpoints': {
            'GET /': 'Документация',
            'GET /health': 'Проверка состояния',
            'POST /separate': 'Разделение аудио',
            'GET /download/{{id}}/{{stem}}': 'Скачивание результата'
        }
    }