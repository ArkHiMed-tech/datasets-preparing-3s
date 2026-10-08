# Diabetes dataset preparation

Проект выполняет подготовку и демонстрацию данных для обучения модели классификации диабета. Проект рассчитан на запуск в Linux и использует относительные пути, вычисляемые от расположения исходного файла.

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

