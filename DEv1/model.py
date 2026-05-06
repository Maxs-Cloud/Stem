import torch
from demucs_infer.pretrained import get_model
from demucs_infer.apply import apply_model
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

class DemucsSeparator:
    """
    Класс для разделения аудио на стемы с помощью предобученной модели Demucs.
    Использует библиотеку demucs-infer.
    """

    def __init__(self, model_name: str = 'htdemucs_ft', device: Optional[str] = None):
        """
        Инициализация разделителя.

        Args:
            model_name: название модели (по умолчанию 'htdemucs_ft').
            device: устройство для инференса ('cuda', 'cpu' или None для автоопределения).
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.sample_rate = 44100  # Demucs ожидает 44100 Гц

        print(f"Загрузка модели {model_name} на {self.device}...")
        self.model = get_model(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.apply_model = apply_model
        print("Модель успешно загружена!")

    def separate(self, audio_path: str, output_dir: Optional[str] = None) -> Dict[str, np.ndarray]:
        """
        Разделение аудиофайла на стемы.

        Args:
            audio_path: путь к аудиофайлу (поддерживаются wav, mp3, flac и др.).
            output_dir: если указан, сохраняет стемы в виде wav-файлов в эту папку.

        Returns:
            Словарь с numpy-массивами стемов:
            {'drums', 'bass', 'other', 'vocals'}.
            Каждый массив имеет форму (2, samples) – стерео.
        """
        print(f"Загрузка аудио: {audio_path}")
        # Загружаем аудио с частотой 44100 Гц, стерео
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=False)

        # Приводим к правильному формату: (2, samples)
        if audio.ndim == 1:
            # Моно → дублируем канал
            audio = np.stack([audio, audio])
        elif audio.ndim == 2 and audio.shape[0] > 2:
            # Многоканальное (например, 5.1) → берём первые два канала
            audio = audio[:2, :]

        # Конвертируем в тензор и добавляем размерность батча
        audio_tensor = torch.from_numpy(audio).float().to(self.device).unsqueeze(0)

        print("Разделение на стемы...")
        with torch.no_grad():
            sources = self.apply_model(
                self.model,
                audio_tensor,
                device=self.device,
                shifts=1,
                split=True,
                overlap=0.25
            )

        # sources имеет форму (1, 4, 2, samples) → (источник, канал, время)
        result = {
            'drums': sources[0, 0].cpu().numpy(),
            'bass': sources[0, 1].cpu().numpy(),
            'other': sources[0, 2].cpu().numpy(),
            'vocals': sources[0, 3].cpu().numpy()
        }

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            for name, audio_array in result.items():
                output_path = output_dir / f"{name}.wav"
                # Транспонируем для soundfile: (samples, channels)
                sf.write(str(output_path), audio_array.T, self.sample_rate)
                print(f"Сохранён стем: {output_path}")

        return result