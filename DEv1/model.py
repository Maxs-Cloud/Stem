import os
import torch
import numpy as np
import soundfile as sf
import warnings
from pathlib import Path
from typing import Dict, List
import librosa
from demucs.pretrained import get_model

warnings.filterwarnings('ignore')

class DemucsSeparator:
    """
    Класс для разделения аудио на стемы с помощью Demucs.
    model_name: название модели Demucs по умолчанию 'htdemucs'
    """

    def __init__(self, model_name: str = "htdemucs_ft", device: str = None):

        self.model_name = model_name
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.sample_rate = 44100  # Demucs работает на 44100 Гц

        # Загружаем модель
        print(f"Загрузка модели {model_name} на {self.device}...")
        from demucs import pretrained
        self.model = get_model(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        print("Модель загружена!")

    def separate(self, audio_path: str, output_dir: str = None) -> Dict[str, np.ndarray]:
        """
        Разделяет аудиофайл на стемы.
        Args:
            audio_path: путь к аудиофайлу
            output_dir: директория для сохранения стемов (если None, возвращает numpy массивы)
        Returns:
            Словарь с стемами: {'vocals': array, 'drums': array, 'bass': array, 'other': array}
        """
        # Загружаем аудио
        print(f"Загрузка аудио: {audio_path}")
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=False)

        # Приводим к правильному формату
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]  # Добавляем размерность каналов
        elif audio.ndim == 2 and audio.shape[0] > 2:
            audio = audio[:2, :]

        audio_tensor = torch.from_numpy(audio).float().to(self.device)

        # Разделение с чанкингом для длинных треков
        with torch.no_grad():
            stems = self._separate_with_chunking(audio_tensor)

        # Конвертируем обратно в numpy
        result = {}
        stem_names = ['vocals', 'drums', 'bass', 'other']

        for i, name in enumerate(stem_names):
            result[name] = stems[i].cpu().numpy()

        # Сохраняем если указана директория
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            for name, audio_array in result.items():
                output_path = output_dir / f"{name}.wav"
                # Транспонируем для soundfile: (samples, channels)
                sf.write(output_path, audio_array.T, self.sample_rate)
                print(f"Сохранен стем: {output_path}")

        return result

    def _separate_with_chunking(self, audio: torch.Tensor, chunk_size: int = 44100 * 30) -> torch.Tensor:
        """
        Разделение с обработкой по чанкам для длинных треков.

        Args:
            audio: входной тензор (channels, samples)
            chunk_size: размер чанка в сэмплах (30 секунд по умолчанию)

        Returns:
            Тензор стемов (4, channels, samples)
        """
        length = audio.shape[-1]

        if length <= chunk_size:
            # Если трек короткий, обрабатываем целиком
            from demucs.apply import apply_model
            stems = apply_model(self.model, audio.unsqueeze(0),
                              shifts=1, split=True, overlap=0.25)[0]
            return stems[:4]  # vocals, drums, bass, other

        # Чанкинг с перекрытием
        overlap = chunk_size // 4
        step = chunk_size - overlap
        stems = []

        for start in range(0, length, step):
            end = min(start + chunk_size, length)
            chunk = audio[:, start:end]

            # Паддинг если последний чанк короче
            if chunk.shape[-1] < chunk_size:
                padding = chunk_size - chunk.shape[-1]
                chunk = torch.nn.functional.pad(chunk, (0, padding))

            # Разделение чанка
            from demucs.apply import apply_model
            chunk_stems = apply_model(self.model, chunk.unsqueeze(0),
                                     shifts=1, split=False, overlap=0.25)[0]

            # Обрезаем паддинг если был
            if padding > 0:
                chunk_stems = chunk_stems[..., :end-start]

            # Применяем веса для плавного перекрытия
            if start > 0 and stems:
                fade_in = torch.linspace(0, 1, overlap).to(self.device)
                fade_out = torch.linspace(1, 0, overlap).to(self.device)

                # Применяем кроссфейд
                for i in range(4):
                    stems[i][..., -overlap:] = stems[i][..., -overlap:] * fade_out
                    chunk_stems[i][..., :overlap] = chunk_stems[i][..., :overlap] * fade_in

            # Собираем результат
            if not stems:
                stems = [chunk_stems[i] for i in range(4)]
            else:
                # Конкатенируем без перекрытия
                for i in range(4):
                    stems[i] = torch.cat([
                        stems[i][..., :-overlap] if start > 0 else stems[i],
                        chunk_stems[i]
                    ], dim=-1)

        return torch.stack(stems)

    def trim_silence_simple(self, audio, sample_rate):
        if audio.ndim > 1:
            signal = audio[0]
        else:
            signal = audio

        trimmed, indices = librosa.effects.trim(signal, top_db=30)

        if audio.ndim > 1:
            return audio[:, indices[0]:indices[1]]
        return trimmed

    def trim_silence(self, audio: np.ndarray, threshold_db: float = -40) -> np.ndarray:
        """
        Обрезка тишины на границах трека.
        Args:
            audio: аудио массив
            threshold_db: порог в децибелах для определения тишины
        """

        energy = librosa.feature.rms(y=audio[0] if audio.ndim > 1 else audio)[0]
        energy_db = librosa.amplitude_to_db(energy, ref=np.max)

        # Находим границы где сигнал выше порога
        mask = energy_db > threshold_db

        if not mask.any():
            return audio

        # Находим индексы начала и конца
        onsets = np.where(mask)[0]
        start_sample = max(0, onsets[0] * 512 - 44100)  # -1 секунда
        end_sample = min(len(audio[0]) if audio.ndim > 1 else len(audio),
                        onsets[-1] * 512 + 44100)  # +1 секунда

        if audio.ndim > 1:
            return audio[:, start_sample:end_sample]
        return audio[start_sample:end_sample]