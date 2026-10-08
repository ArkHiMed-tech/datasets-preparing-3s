# Diabetes dataset preparation

Проект выполняет подготовку и демонстрацию данных для обучения модели классификации диабета. Он использует относительные пути и позволяет запускать подготовку, обучение и проверку с помощью `uv` и Bash.

## Краткое описание проекта

Проект очищает и преобразует датасет диабета, обучает модель LightGBM через пайплайн с препроцессинговыми шагами, сохраняет модель и выполняет валидацию на отдельных метриках. Для каждого запуска результаты validation сохраняются в каталоге `results/experiments/`.

## Требования к окружению

- Python `>=3.10,<3.11`.
- `uv` для создания и запуска окружения.
- На Linux и WSL для установки `uv` используйте официальный скрипт установки:

  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

- На Windows можно установить `uv` через Winget:

  ```powershell
  winget install --id=astral-sh.uv -e
  ```

## Установка зависимостей

Из корня проекта выполните:

```bash
uv sync --locked
```

Команда создаёт окружение из `pyproject.toml` и `uv.lock`. При необходимости окружение можно пересоздать:

```bash
uv sync --reinstall --locked
```

## Запуск проекта

Подготовьте данные, обучите модель и выполните валидацию:

```bash
uv run python src/experiments/prepare_datasets.py \
  --model-name diabetes_best_model_pipeline \
  --data-path data/raw \
  --random-seed 42 \
  --epochs 100 \
  --output-dir .

uv run python src/experiments/validate_pipeline.py \
  --model-name diabetes_best_model_pipeline \
  --output-dir .
```

Для автоматического запуска обоих этапов используйте Bash-скрипт:

```bash
bash scripts/run_project.sh
```

## Описание Bash-скрипта

Скрипт `scripts/run_project.sh` выполняет:

1. Проверку существования `pyproject.toml` и входного датасета.
2. Проверку корректности `--random-seed` и `--epochs`.
3. Поиск `uv` в текущем окружении или в Windows-путях WSL.
4. Синхронизацию зависимостей через `uv sync`.
5. Запуск `prepare_datasets.py` с параметрами модели, датасета, seed, количество эпох и каталогом вывода.
6. Запуск `validate_pipeline.py` и сохранение метрик в `results/experiments/<model-name>/`.

Доступные параметры:

```text
--model-name NAME
--data-path PATH
--random-seed INTEGER
--epochs INTEGER
--output-dir PATH
-h, --help
```

Например:

```bash
bash scripts/run_project.sh \
  --model-name diabetes_best_model_pipeline \
  --data-path data/raw \
  --random-seed 42 \
  --epochs 100 \
  --output-dir .
```

## Структура проекта

- `src/`: исходный код приложения и экспериментальных скриптов.
- `src/notebooks/`: Jupyter-ноутбук с подготовкой, обучением и проверкой модели.
- `src/experiments/`: воспроизводимые скрипты подготовки и экспериментов.
- `data/raw/`: исходные CSV-файлы.
- `data/processed/`: очищённый и преобразованный датасет.
- `models/`: сохранённые обученные модели.
- `results/experiments/`: журналы, результаты и временные метрики экспериментов.
- `config/`: конфигурационные и вспомогательные файлы.

## Запуск в Linux с uv

1. Установите `uv` и синхронизируйте окружение:

   ```bash
   uv sync
   ```

2. Подготовьте данные:

   ```bash
   uv run python src/experiments/prepare_datasets.py
   ```

3. Запустите web-приложение:

   ```bash
   uv run streamlit run src/app.py
   ```

4. Для запуска ноутбука откройте `src/notebooks/main.ipynb` в Jupyter Notebook или VS Code:

   ```bash
   uv run jupyter notebook
   ```

## Важные замечания

- Основной источник зависимостей — `pyproject.toml`.
- `uv.lock` фиксирует версии зависимостей и создаёт воспроизводимое окружение.
- `.venv` не отслеживается Git.
- `config/requirements.txt` и `config/.python-version` оставлены только для совместимости с существующими скриптами.
- Скрипт подготовки читает исходные CSV из `data/raw/` и сохраняет итоговый датасет в `data/processed/diabetes.csv`.
- Сохранённая модель записывается в `models/diabetes_best_model_pipeline.pkl`.
- При необходимости повторного запуска существующие исходные файлы не удаляются.

## Источники датасета

1. https://www.kaggle.com/datasets/johndasilva/diabetes
2. https://www.kaggle.com/datasets/ehababoelnaga/diabetes-dataset
3. https://www.kaggle.com/datasets/salihacur/diabetes
4. https://www.kaggle.com/datasets/mustafaoz158/diabetes
5. https://www.kaggle.com/datasets/arezalo/diabetes
6. https://www.kaggle.com/datasets/sztuanakurun/diabates
7. https://www.kaggle.com/datasets/fathyfathysahlool/diabetes
8. https://www.kaggle.com/datasets/figolm10/diabetes

