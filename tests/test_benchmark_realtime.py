import unittest

from src.benchmark_realtime import estimated_fps


class RealtimeBenchmarkTests(unittest.TestCase):
    def test_estimated_fps_formula(self):
        self.assertAlmostEqual(estimated_fps(25.0), 40.0)
        self.assertEqual(estimated_fps(0.0), 0.0)


if __name__ == "__main__":
    unittest.main()
