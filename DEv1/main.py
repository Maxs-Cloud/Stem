#!/usr/bin/env python3
import musdb
import uvicorn
from torch.xpu import device

from metrics import MetricsEvaluator
from model import DemucsSeparator

def main():
    # 1. Загружаем тестовый датасет (без скачивания, должна быть папка DEv1/db_test/test)

    db = musdb.DB(
        root=r'C:\Users\marhr\PycharmProjects\PythonProject5\DEv1\db_test',    # путь к папке, содержащей test/
        subsets='test',
        download=False,
        is_wav=False
    )

    # 2. Создаём разделитель с локальным файлом весов
    separator = DemucsSeparator(device='cuda')

    # 3. Оценка метрик на 10 треках
    evaluator = MetricsEvaluator(db, show_spectogramm=False)
    summary = evaluator.evaluate_dataset(separator, num_tracks=1)

if __name__ == "__main__":
   #main()
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)