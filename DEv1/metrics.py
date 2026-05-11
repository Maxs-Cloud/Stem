import museval
import numpy as np
import musdb
from pathlib import Path
from typing import Dict, Optional
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import tempfile
import soundfile as sf
import librosa
import librosa.display

warnings.filterwarnings('ignore')

class MetricsEvaluator:
    """
    Класс для оценки качества разделения на датасете MUSDB18.
    Принимает готовый объект musdb.DB и флаг показа спектрограмм.
    """

    def __init__(self, db, show_spectogramm: bool = False):
        """
        Args:
            db: объект musdb.DB, загруженный с subsets='test' и download=True
            show_spectogramm: показывать спектрограммы первого трека после оценки
        """
        self.db = db
        self.show_spectogramm = show_spectogramm
        self.stem_names = ['vocals', 'drums', 'bass', 'other']

    def evaluate_track(self, reference: Dict[str, np.ndarray],
                       estimated: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """Оценка метрик для одного трека."""
        ref_list = []
        est_list = []

        for name in self.stem_names:
            ref = reference[name]
            est = estimated[name]
            if est.ndim == 2 and est.shape[0] == 2 and est.shape[1] > 2:
                est = est.T
            ref_list.append(ref)
            est_list.append(est)

        ref_matrix = np.stack(ref_list, axis=0)
        est_matrix = np.stack(est_list, axis=0)

        sdr, isr, sir, sar, _ = museval.metrics.bss_eval(
            ref_matrix, est_matrix, compute_permutation=False
        )

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
        После вычисления метрик сразу строит графики.
        """
        db = self.db
        all_results = []

        # сохраняем данные первого трека для возможных спектрограмм
        first_estimated = None
        first_reference = None
        first_track_name = ""

        for idx, track in enumerate(tqdm(db[:num_tracks], desc="Оценка треков")):
            reference = {
                'vocals': track.targets['vocals'].audio,
                'drums': track.targets['drums'].audio,
                'bass': track.targets['bass'].audio,
                'other': track.audio - track.targets['vocals'].audio
                         - track.targets['drums'].audio
                         - track.targets['bass'].audio
            }

            mix = track.audio
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                sf.write(tmp.name, mix, 44100)
                tmp_path = tmp.name

            estimated = separator.separate(tmp_path)
            Path(tmp_path).unlink()

            track_results = self.evaluate_track(reference, estimated)

            for stem_name, metrics in track_results.items():
                metrics['track'] = track.name
                metrics['stem'] = stem_name
                all_results.append(metrics)

            # запоминаем первый трек для спектрограмм
            if idx == 0:
                first_estimated = estimated
                first_reference = reference
                first_track_name = track.name

        if not all_results:
            print("⚠️ Нет данных для оценки. Проверьте, что датасет загружен правильно.")
            return pd.DataFrame()

        # Усредняем метрики по стемам
        df = pd.DataFrame(all_results)
        summary = df.groupby('stem')[['SDR', 'SIR', 'SAR', 'ISR']].mean().reset_index()

        # ----- ВИЗУАЛИЗАЦИЯ (встроена прямо сюда) -----
        if self.show_spectogramm and first_estimated is not None:
            self._plot_with_spectrograms(summary, first_estimated,
                                         first_reference, first_track_name)
        else:
            self._plot_metrics_only(summary)

        # Таблица в консоли
        print("\n" + "=" * 60)
        print("СРЕДНИЕ МЕТРИКИ ПО СТЕМАМ (dB)")
        print("=" * 60)
        print(summary.to_string(index=False))
        print("=" * 60)

        return summary

    # -----------------------------------------------------------------
    # Внутренние методы визуализации
    # -----------------------------------------------------------------
    def _plot_metrics_only(self, df: pd.DataFrame):
        """Только столбчатые диаграммы."""
        metrics = ['SDR', 'SIR', 'SAR', 'ISR']
        stem_colors = {
            'vocals': '#E74C3C', 'drums': '#3498DB',
            'bass': '#2ECC71', 'other': '#9B59B6'
        }

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('BSSEval Metrics (averaged over dataset)',
                     fontsize=16, fontweight='bold')
        for idx, metric in enumerate(metrics):
            ax = axes.flatten()[idx]
            self._draw_single_metric(ax, df, metric, stem_colors)

        plt.tight_layout()
        plt.savefig('metrics_summary.png', dpi=150, bbox_inches='tight', facecolor='white')
        plt.show()
        print("График сохранён: metrics_summary.png")

    def _plot_with_spectrograms(self, df, estimated, reference, track_name):
        metrics = ['SDR', 'SIR', 'SAR', 'ISR']
        stem_colors = {'vocals': '#E74C3C', 'drums': '#3498DB',
                       'bass': '#2ECC71', 'other': '#9B59B6'}
        stem_cmaps = {'vocals': 'Oranges', 'drums': 'Blues',
                      'bass': 'Greens', 'other': 'Purples'}

        n_stems = len(self.stem_names)
        has_ref = reference is not None

        # Определяем количество колонок и их пропорции
        if has_ref:
            n_cols = 4  # est | ref | gap | metrics
            width_ratios = [1, 1, 0.2, 1.5]
            ref_col = 1
            gap_col = 2
            metric_start_col = 3
        else:
            n_cols = 3  # est | gap | metrics
            width_ratios = [1, 0.2, 1.5]
            ref_col = None
            gap_col = 1
            metric_start_col = 2

        fig = plt.figure(figsize=(6 + 4 * n_cols, 12))
        gs = fig.add_gridspec(n_stems, n_cols,
                              width_ratios=width_ratios,
                              hspace=0.4, wspace=0.3)

        # Спектрограммы (левая часть)
        for i, stem_name in enumerate(self.stem_names):
            ax_est = fig.add_subplot(gs[i, 0])
            est_audio = estimated[stem_name]
            if est_audio.ndim > 1:
                est_audio = np.mean(est_audio, axis=0)
            mel_est = librosa.feature.melspectrogram(y=est_audio, sr=44100,
                                                     n_mels=128, fmax=8000)
            mel_est_db = librosa.power_to_db(mel_est, ref=np.max)
            img = librosa.display.specshow(
                mel_est_db, sr=44100, x_axis='time', y_axis='mel',
                ax=ax_est, cmap=stem_cmaps.get(stem_name, 'magma'), fmax=8000)
            ax_est.set_title(f'{stem_name.upper()} (est)', fontsize=10, fontweight='bold')
            if i == n_stems - 1:
                ax_est.set_xlabel('Time')

            if has_ref:
                ax_ref = fig.add_subplot(gs[i, ref_col])
                ref_audio = reference[stem_name]
                if ref_audio.ndim > 1:
                    ref_audio = np.mean(ref_audio, axis=1)
                mel_ref = librosa.feature.melspectrogram(y=ref_audio, sr=44100,
                                                         n_mels=128, fmax=8000)
                mel_ref_db = librosa.power_to_db(mel_ref, ref=np.max)
                librosa.display.specshow(
                    mel_ref_db, sr=44100, x_axis='time', y_axis='mel',
                    ax=ax_ref, cmap=stem_cmaps.get(stem_name, 'magma'), fmax=8000)
                ax_ref.set_title(f'{stem_name.upper()} (ref)', fontsize=10, fontweight='bold')
                if i == n_stems - 1:
                    ax_ref.set_xlabel('Time')

        # Colorbar для спектрограмм (используем последнюю ось est)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        fig.colorbar(img, cax=cbar_ax, format='%+2.0f dB')

        # Метрики (правая часть)
        for idx, metric in enumerate(metrics):
            ax = fig.add_subplot(gs[idx, metric_start_col])
            self._draw_single_metric(ax, df, metric, stem_colors)

        fig.suptitle(f'BSSEval Metrics & Spectrograms — {track_name}',
                     fontsize=18, fontweight='bold', y=1.02)
        plt.savefig('metrics_summary.png', dpi=150, bbox_inches='tight', facecolor='white')
        plt.show()
        print("График со спектрограммами сохранён: metrics_summary.png")
    @staticmethod
    def _draw_single_metric(ax, df, metric, stem_colors):
        """Рисует один столбчатый график для заданной метрики."""
        df_sorted = df.sort_values(metric, ascending=False)
        colors = [stem_colors.get(stem, '#95A5A6') for stem in df_sorted['stem']]

        bars = ax.bar(df_sorted['stem'], df_sorted[metric],
                      color=colors, edgecolor='white', linewidth=1.2)

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