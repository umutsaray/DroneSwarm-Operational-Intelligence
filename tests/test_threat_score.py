import unittest

from src.threat_score import calculate_a_swarm, calculate_threat_score, risk_level


class ThreatScoreTests(unittest.TestCase):
    def test_paper_formula(self):
        config = {
            "threat_score": {
                "alpha": 0.2,
                "beta": 0.3,
                "gamma": 0.5,
                "N_max": 4,
                "D_max": 100,
                "d_max": 1,
                "protected_region_center": [0.5, 0.5],
                "min_swarm_area": 1e-6,
            }
        }
        boxes = [(0, 0.5, 0.5, 0.2, 0.2)]
        metrics = calculate_threat_score(boxes, config)
        self.assertAlmostEqual(metrics["N_norm"], 0.25)
        self.assertAlmostEqual(metrics["A_swarm"], 0.04)
        self.assertAlmostEqual(metrics["D"], 0.25)
        self.assertAlmostEqual(metrics["P"], 1.0)
        self.assertAlmostEqual(metrics["final_TS"], 0.625)

    def test_risk_boundaries(self):
        self.assertEqual(risk_level(0.0), "Low")
        self.assertEqual(risk_level(0.249), "Low")
        self.assertEqual(risk_level(0.25), "Medium")
        self.assertEqual(risk_level(0.50), "High")
        self.assertEqual(risk_level(0.75), "Critical")
        self.assertEqual(risk_level(1.2), "Critical")

    def test_a_swarm_two_boxes(self):
        boxes = [(0, 0.2, 0.5, 0.2, 0.2), (0, 0.8, 0.5, 0.2, 0.2)]
        self.assertAlmostEqual(calculate_a_swarm(boxes), 0.16)


if __name__ == "__main__":
    unittest.main()
