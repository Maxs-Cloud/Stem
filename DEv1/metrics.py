import museval
import numpy as np
import musdb
from pathlib import Path
from typing import Dict
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import tempfile
import soundfile as sf
import librosa.display

warnings.filterwarnings('ignore')

class MetricsEvaluator:
    """
    Класс для оценки качества разделения на датасете MUSDB18.
    Принимает готовый объект musdb.DB.
    """

    def __init__(self, db):
        """
        Args:
            db: объект musdb.DB, загруженный с subsets='test' и download=True
        """
        self.db = db
        self.stem_names = ['vocals', 'drums', 'bass', 'other']

    def evaluate_track(self, reference: Dict[str, np.ndarray],
                       estimated: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """
        Оценка метрик для одного трека.

        Args:
            reference: эталонные стемы в формате (samples, channels)
            estimated: предсказанные стемы в формате (channels, samples)
        """
        # Приводим оба к формату (sources, samples, channels)
        ref_list = []
        est_list = []

        for name in self.stem_names:
            # Эталон уже в правильном формате (samples, channels)
            ref = reference[name]

            # Предсказание нужно транспонировать: (channels, samples) → (samples, channels)
            est = estimated[name]
            if est.ndim == 2 and est.shape[0] == 2 and est.shape[1] > 2:
                est = est.T

            ref_list.append(ref)
            est_list.append(est)

        # Собираем матрицы (4, samples, 2)
        ref_matrix = np.stack(ref_list, axis=0)
        est_matrix = np.stack(est_list, axis=0)

        # Вычисляем метрики
        sdr, isr, sir, sar, _ = museval.metrics.bss_eval(
            ref_matrix,
            est_matrix,
            compute_permutation=False
        )

        # Собираем результат: медиана по всем временным окнам
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
        Оценка на num_tracks треках из MUSDB18.

        Args:
            separator: экземпляр DemucsSeparator с методом separate()
            num_tracks: количество треков для оценки

        Returns:
            DataFrame с усреднёнными метриками по стемам
        """
        db = self.db
        all_results = []

        for track in tqdm(db[:num_tracks], desc="Оценка треков"):
            # 1. Эталонные стемы из датасета — формат (samples, channels)
            reference = {
                'vocals': track.targets['vocals'].audio,
                'drums': track.targets['drums'].audio,
                'bass': track.targets['bass'].audio,
                'other': track.audio - track.targets['vocals'].audio
                         - track.targets['drums'].audio
                         - track.targets['bass'].audio
            }

            # 2. Сохраняем микс во временный WAV
            mix = track.audio  # (samples, channels)
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                sf.write(tmp.name, mix, 44100)
                tmp_path = tmp.name

            # 3. Разделяем — получаем стемы в формате (channels, samples)
            estimated = separator.separate(tmp_path)

            # 4. Сравниваем (транспонирование внутри evaluate_track)
            track_results = self.evaluate_track(reference, estimated)

            # 5. Удаляем временный файл
            Path(tmp_path).unlink()

            # 6. Добавляем метрики в общий список
            for stem_name, metrics in track_results.items():
                metrics['track'] = track.name
                metrics['stem'] = stem_name
                all_results.append(metrics)

        # Защита от пустого списка
        if not all_results:
            print("⚠️ Нет данных для оценки. Проверьте, что датасет загружен правильно.")
            return pd.DataFrame()

        # Усредняем метрики по стемам
        df = pd.DataFrame(all_results)
        summary = df.groupby('stem')[['SDR', 'SIR', 'SAR', 'ISR']].mean().reset_index()

        # Визуализация
        self.plot_metrics(summary)

        return summary

    def plot_metrics(self, df: pd.DataFrame, estimated=None, reference=None, track_name=""):

        metrics = ['SDR', 'SIR', 'SAR', 'ISR']
        stem_colors = {
        'vocals': '#E74C3C',
        'drums':  '#3498DB',
        'bass':   '#2ECC71',
        'other':  '#9B59B6'
    }
        stem_cmaps = {
        'vocals': 'Oranges',
        'drums': 'Blues',
        'bass': 'Greens',
        'other': 'Purples'
    }

    # Определяем layout
        show_spectrograms = False
        n_cols_spect = 2 if reference else 1
        n_rows_spect = 4 if show_spectrograms else 0

        if show_spectrograms:
            # Большая фигура: слева спектрограммы, справа метрики
            fig = plt.figure(figsize=(20, 12))
            gs = fig.add_gridspec(4, 3, width_ratios=[1, 0.5, 1.5], hspace=0.4, wspace=0.3)
        else:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes_flat = axes.flatten()

        fig.suptitle(f'BSSEval Metrics & Spectrograms — {track_name}',
                     fontsize=18, fontweight='bold', y=1.02)

        # ==================== СПЕКТРОГРАММЫ (слева) ====================
        if show_spectrograms:
            for i, stem_name in enumerate(self.stem_names):
                # Предсказанный стем
                ax_est = fig.add_subplot(gs[i, 0])
                est_audio = estimated[stem_name]
                if est_audio.ndim > 1:
                    est_audio = np.mean(est_audio, axis=0)

                mel_est = librosa.feature.melspectrogram(y=est_audio, sr=44100, n_mels=128, fmax=8000)
                mel_est_db = librosa.power_to_db(mel_est, ref=np.max)

                img = librosa.display.specshow(
                    mel_est_db, sr=44100, x_axis='time', y_axis='mel',
                    ax=ax_est, cmap=stem_cmaps.get(stem_name, 'magma'), fmax=8000
                )
                ax_est.set_title(f'{stem_name.upper()} (est)', fontsize=10, fontweight='bold')
                if i == 3:
                    ax_est.set_xlabel('Time')

                # Эталонный стем
                if reference:
                    ax_ref = fig.add_subplot(gs[i, 1])
                    ref_audio = reference[stem_name]
                    if ref_audio.ndim > 1:
                        ref_audio = np.mean(ref_audio, axis=1)

                    mel_ref = librosa.feature.melspectrogram(y=ref_audio, sr=44100, n_mels=128, fmax=8000)
                    mel_ref_db = librosa.power_to_db(mel_ref, ref=np.max)

                    librosa.display.specshow(
                        mel_ref_db, sr=44100, x_axis='time', y_axis='mel',
                        ax=ax_ref, cmap=stem_cmaps.get(stem_name, 'magma'), fmax=8000
                    )
                    ax_ref.set_title(f'{stem_name.upper()} (ref)', fontsize=10, fontweight='bold')
                    if i == 3:
                        ax_ref.set_xlabel('Time')

            # Colorbar для спектрограмм
            cbar_ax = fig.add_axes([0.42, 0.15, 0.015, 0.7])
            fig.colorbar(img, cax=cbar_ax, format='%+2.0f dB')

            # Метрики (справа)
            for idx, metric in enumerate(metrics):
                ax = fig.add_subplot(gs[idx, 2])
                _plot_single_metric(ax, df, metric, stem_colors)
        else:
            # Только метрики
            for idx, metric in enumerate(metrics):
                ax = axes_flat[idx]
                _plot_single_metric(ax, df, metric, stem_colors)

        plt.savefig('metrics_summary.png', dpi=150, bbox_inches='tight', facecolor='white')
        plt.show()
        print("График сохранён: metrics_summary.png")

        # Таблица в консоли
        print("\n" + "=" * 60)
        print("СРЕДНИЕ МЕТРИКИ ПО СТЕМАМ (dB)")
        print("=" * 60)
        print(df.to_string(index=False))
        print("=" * 60)


def _plot_single_metric(ax, df, metric, stem_colors):
    """Вспомогательная функция для отрисовки одной метрики."""
    df_sorted = df.sort_values(metric, ascending=False)
    colors = [stem_colors.get(stem, '#95A5A6') for stem in df_sorted['stem']]

    bars = ax.bar(df_sorted['stem'], df_sorted[metric], color=colors, edgecolor='white', linewidth=1.2)

    ax.yaxis.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.set_title(f'{metric}', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score (dB)', fontsize=11)
    ax.set_ylim(min(df[metric].min() - 3, -10), max(df[metric].max() + 3, 20))

    for bar, value in zip(bars, df_sorted[metric]):
        color = 'darkred' if value < 0 else 'darkgreen'
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5 if value >= 0 else bar.get_height() - 1.5,
            f'{value:.1f}',
            ha='center', va='bottom' if value >= 0 else 'top',
            fontsize=11, fontweight='bold', color=color
        )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)