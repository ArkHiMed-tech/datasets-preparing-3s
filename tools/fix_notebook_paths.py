import json
from pathlib import Path

NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "src" / "notebooks" / "main.ipynb"
notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))

for cell in notebook["cells"]:
    if cell.get("cell_type") != "code":
        continue
    source = "".join(cell.get("source", []))
    if "import pandas as pd" in source:
        source = source.replace(
            "import pandas as pd",
            "from pathlib import Path\n\nimport pandas as pd",
            1,
        )
        source = source.replace(
            "import seaborn as sns",
            "import seaborn as sns\n\nPROJECT_ROOT = Path.cwd()\n"
            "while not (PROJECT_ROOT / 'data' / 'raw').exists():\n"
            "    PROJECT_ROOT = PROJECT_ROOT.parent\n"
            "RAW_DATA_DIR = PROJECT_ROOT / 'data' / 'raw'\n"
            "PROCESSED_DATA_PATH = PROJECT_ROOT / 'data' / 'processed' / 'diabetes.csv'\n"
            "MODELS_DIR = PROJECT_ROOT / 'models'",
            1,
        )
    source = source.replace(
        "files = ['diabetes (1).csv', 'diabetes (2).csv', 'diabetes (3).csv', "
        "'diabetes (4).csv', 'diabetes (5).csv', 'diabetes (6).csv', "
        "'diabetes (7).csv', 'diabetes (8).csv']",
        "files = [RAW_DATA_DIR / f'diabetes ({number}).csv' for number in range(1, 9)]",
    )
    source = source.replace(
        "df.to_csv('diabetes.csv', index=False)",
        "PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)\n"
        "df.to_csv(PROCESSED_DATA_PATH, index=False)",
    )
    source = source.replace(
        "filename = 'diabetes_best_model_pipeline.pkl'",
        "model_path = MODELS_DIR / 'diabetes_best_model_pipeline.pkl'",
    )
    source = source.replace(
        "joblib.dump(pipeline, filename)",
        "joblib.dump(pipeline, model_path)",
    )
    source = source.replace(
        "joblib.load('diabetes_best_model_pipeline.pkl')",
        "joblib.load(model_path)",
    )
    source = source.replace(
        "pd.read_csv('diabetes.csv')",
        "pd.read_csv(PROCESSED_DATA_PATH)",
    )
    cell["source"] = source.splitlines(keepends=True)

NOTEBOOK_PATH.write_text(
    json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
)
print(f"Updated {NOTEBOOK_PATH}")
