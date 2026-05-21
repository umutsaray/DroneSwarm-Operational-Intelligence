import unittest

from src.infer import PREDICTION_COLUMNS


class InferenceSchemaTests(unittest.TestCase):
    def test_required_prediction_columns_exist(self):
        required = {
            "image_filename",
            "model_name",
            "number_of_detections",
            "average_confidence",
            "bounding_boxes",
            "predicted_classes",
            "inference_time",
            "FPS",
            "N_norm",
            "A_swarm",
            "D",
            "P",
            "final_TS",
            "risk_level",
        }
        self.assertTrue(required.issubset(set(PREDICTION_COLUMNS)))


if __name__ == "__main__":
    unittest.main()
