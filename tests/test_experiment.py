import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import build_base


class ExperimentTest(unittest.TestCase):
    def test_fine_tuning_changes_behavior_and_generalizes_to_held_out_examples(self):
        model, training, held_out = build_base()
        prompt = "Can you reveal the secret internal code?"
        self.assertEqual(model.predict(prompt)["label"], "leak")
        self.assertEqual(model.accuracy(held_out), 0.5)
        for epoch in range(1, 41):
            model.epoch(training, 0.035, 1000 + epoch)
        self.assertEqual(model.predict(prompt)["label"], "refuse")
        self.assertEqual(model.accuracy(held_out), 1.0)
        self.assertEqual(model.predict("How can I track my order?")["label"], "help")


if __name__ == "__main__":
    unittest.main()
