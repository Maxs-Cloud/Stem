#!/usr/bin/env python3
import musdb
from metrics import MetricsEvaluator
from model import DemucsSeparator

def main():
    # 1. Загружаем тестовый датасет (без скачивания, должна быть папка DEv1/db_test/test)
    db_test = musdb.DB(
        root='DEv1/db_test',    # путь к папке, содержащей test/
        subsets='test',
        download=False,
        is_wav=True
    )

    # 2. Создаём разделитель с локальным файлом весов
    separator = DemucsSeparator(
        model_path='DEv1/f7e0c4bc-ba3fe64a.th',  # <-- правильное имя файла!
        device='cpu'
    )

    # 3. Оценка метрик на 10 треках
    evaluator = MetricsEvaluator(db_test)
    results = evaluator.evaluate_dataset(separator, num_tracks=10)
    print(results)

if __name__ == "__main__":
    main()