import torch
from demucs_infer.htdemucs import HTDemucs  # архитектура модели
from demucs_infer.apply import apply_model
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

class DemucsSeparator:
    """
    Класс для разделения аудио с загрузкой модели из локального файла.
    """

    def __init__(self, model_path=None, device=None):
        """
        Args:
            model_path: путь к файлу весов (например, './Dev1/f7e0c4bc-ba3fe64a.th').
                        Если не указан, ищет в стандартном кэше PyTorch.
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.sample_rate = 44100

        # Загружаем архитектуру HTDemucs с параметрами по умолчанию (4 источника)
        self.model = HTDemucs(sources=['drums', 'bass', 'other', 'vocals'])
        self.model.to(self.device)
        self.model.eval()

        # Если передан путь, загружаем веса из локального файла
        if model_path and Path(model_path).exists():
            print(f"Загрузка весов из {model_path} ...")
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict, strict=True)
            print("Веса успешно загружены.")
        else:
            # Иначе используем стандартный механизм (интернет)
            from demucs_infer.pretrained import get_model
            print(f"Загрузка модели htdemucs_ft через интернет...")
            self.model = get_model('htdemucs_ft')
            self.model.to(self.device)
            self.model.eval()

        self.apply_model = apply_model
        print("Модель готова!")

    def separate(self, audio_path: str, output_dir: str = None) -> Dict[str, np.ndarray]:
        """Разделение аудио на стемы (drums, bass, other, vocals)."""
        print(f"Загрузка аудио: {audio_path}")
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=False)

        if audio.ndim == 1:
            audio = np.stack([audio, audio])   # моно -> псевдо-стерео
        elif audio.ndim == 2 and audio.shape[0] > 2:
            audio = audio[:2, :]

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
                sf.write(str(output_path), audio_array.T, self.sample_rate)
                print(f"Сохранен стем: {output_path}")

        return result