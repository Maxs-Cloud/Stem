import os
import urllib.request

# 1. Прямая ссылка на файл весов модели
model_url = "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/955717e8-8726e21a.th"
# 2. Ссылка на YAML-файл с конфигурацией (паспорт модели)
yaml_url = "https://raw.githubusercontent.com/facebookresearch/demucs/main/htdemucs.yaml"

# 3. Определяем папку, где demucs будет искать локальные модели
local_repo_dir = os.path.expanduser("~/.cache/torch/hub/local_repo")
os.makedirs(local_repo_dir, exist_ok=True)

# 4. Скачиваем и сохраняем файлы
print("Скачивание файла весов...")
urllib.request.urlretrieve(model_url, os.path.join(local_repo_dir, "955717e8-8726e21a.th"))
print("Скачивание YAML-файла...")
urllib.request.urlretrieve(yaml_url, os.path.join(local_repo_dir, "htdemucs.yaml"))

print("Готово! Теперь можно запускать основной код.")