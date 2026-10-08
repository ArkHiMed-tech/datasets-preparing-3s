import json
import tempfile
import unittest
from pathlib import Path

from src.experiments.validate_pipeline import save_validation_results


class SaveValidationResultsTests(unittest.TestCase):
    def test_creates_results_directory_and_reports(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            model_name = "test_model"
            metrics = {"accuracy": 0.9, "roc_auc": 0.8}

            result_dir = save_validation_results(
                output_dir=output_dir,
                model_name=model_name,
                metrics=metrics,
                dataset_shape=(10, 9),
                prediction_count=10,
            )

            self.assertTrue((result_dir / "metrics.json").is_file())
            self.assertTrue((result_dir / "validation_report.txt").is_file())
            self.assertEqual(
                json.loads((result_dir / "metrics.json").read_text(encoding="utf-8")),
                metrics,
            )
            self.assertIn(
                "accuracy: 0.9000",
                (result_dir / "validation_report.txt").read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
