import museval
import numpy as np
import musdb
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

class MetricsEvaluator:
    """
    Класс для оценки качества разделения на датасете MUSDB18.
    """
    def __init__(self, musdb_path: str = None):

        self.musdb_path = musdb_path
        self.stem_names = ['vocals', 'drums', 'bass', 'other']

    def evaluate_track(self, reference: Dict[str, np.ndarray],
                       estimated: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """
        Оценка метрик для одного трека.
        arguments:
            reference: эталонные стемы
            estimated: предсказанные стемы
        """
        # Преобразуем в формат (sources, samples, channels)
        ref_matrix = np.stack([reference[name] for name in self.stem_names], axis=0)
        est_matrix = np.stack([estimated[name] for name in self.stem_names], axis=0)

        # Транспонируем если нужно
        if ref_matrix.ndim == 3 and ref_matrix.shape[-1] == 2:
            ref_matrix = ref_matrix.transpose(0, 2, 1)
        if est_matrix.ndim == 3 and est_matrix.shape[-1] == 2:
            est_matrix = est_matrix.transpose(0, 2, 1)

        # Вычисляем метрики
        sdr, isr, sir, sar, _ = museval.metrics.bss_eval(
            ref_matrix,
            est_matrix,
            compute_permutation=False
        )

        # Собираем результаты
        results = {}
        for i, name in enumerate(self.stem_names):
            results[name] = {
                'SDR': float(np.nanmedian(sdr[i])),
                'SIR': float(np.nanmedian(sir[i])),
                'SAR': float(np.nanmedian(sar[i])),
                'ISR': float(np.nanmedian(isr[i]))
            }

        return results

    def evaluate_dataset(self, separator, num_tracks: int = 10) -> pd.DataFrame:
        """
        arguments:
            separator: экземпляр DemucsSeparator
            num_tracks: количество треков для оценки
        """

        if not self.musdb_path:
            raise ValueError("Укажите путь к MUSDB18 через musdb_path")

        print(f"Загрузка MUSDB18 из {self.musdb_path}")
        db = musdb.DB(root=self.musdb_path, subsets='test', is_wav=True)

        all_results = []

        for track in tqdm(db[:num_tracks], desc="Оценка треков"):
            # Получаем эталонные стемы
            reference = {
                'vocals': track.targets['vocals'].audio.T,
                'drums': track.targets['drums'].audio.T,
                'bass': track.targets['bass'].audio.T,
                'other': track.audio.T - track.targets['vocals'].audio.T -
                         track.targets['drums'].audio.T - track.targets['bass'].audio.T
            }

            audio_path = Path(track.path)
            estimated = separator.separate(str(audio_path))

            track_results = self.evaluate_track(reference, estimated) #оценка

            # Добавляем в общий список
            for stem_name, metrics in track_results.items():
                metrics['track'] = track.name
                metrics['stem'] = stem_name
                all_results.append(metrics)

        # Создаем DataFrame
        df = pd.DataFrame(all_results)

        # Вычисляем средние значения
        summary = df.groupby('stem').mean().reset_index()

        # Визуализация
        self.plot_metrics(summary)

        return summary

    def plot_metrics(self, df: pd.DataFrame):

        metrics = ['SDR', 'SIR', 'SAR', 'ISR']
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            bars = ax.bar(df['stem'], df[metric])
            ax.set_title(f'{metric} по стемам (dB)')
            ax.set_ylabel('dB')
            ax.set_ylim(-10, 20)

            # Добавляем значения над столбцами
            for bar, value in zip(bars, df[metric]):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f'{value:.1f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig('metrics_summary.png', dpi=150)
        print("График сохранен: metrics_summary.png")

        # Вывод таблицы
        print("\n" + "=" * 60)
        print("СРЕДНИЕ МЕТРИКИ ПО СТЕМАМ (dB)")
        print("=" * 60)
        print(df.to_string(index=False))
        print("=" * 60)