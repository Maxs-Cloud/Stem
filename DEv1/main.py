#!/usr/bin/env python3
"""
Скрипт для оценки метрик на MUSDB18.
"""
import argparse
#from model import DemucsSeparator
#from metrics import MetricsEvaluator

def main():
    parser = argparse.ArgumentParser(description='Оценка качества разделения аудио')
    parser.add_argument('--musdb-path', type=str, required=True,
                       help='Путь к датасету MUSDB18')
    parser.add_argument('--num-tracks', type=int, default=10,
                       help='Количество треков для оценки (по умолчанию: 10)')
    parser.add_argument('--model', type=str, default='htdemucs',
                       help='Модель Demucs (по умолчанию: htdemucs)')
    parser.add_argument('--device', type=str, default=None,
                       help='Устройство: cuda или cpu (по умолчанию: авто)')

    args = parser.parse_args()

    # Инициализируем разделитель
    print("Инициализация модели...")
    separator = DemucsSeparator(
        model_name=args.model,
        device=args.device
    )

    # Создаем эвалюатор
    evaluator = MetricsEvaluator(musdb_path=args.musdb_path)

    # Запускаем оценку
    print(f"\nОценка на {args.num_tracks} треках из MUSDB18...")
    results = evaluator.evaluate_dataset(separator, num_tracks=args.num_tracks)

    print("\nОценка завершена!")
    print(f"Результаты сохранены в metrics_summary.png")

if __name__ == "__main__":
  #  main()
   separator = DemucsSeparator(model_name='htdemucs', device=device)# Создаём эвалюатор, передавая уже готовый db_test
   evaluator = MetricsEvaluator(db_test)

# Оценка на 10 треках
   results = evaluator.evaluate_dataset(separator, num_tracks=10)
   print(results)