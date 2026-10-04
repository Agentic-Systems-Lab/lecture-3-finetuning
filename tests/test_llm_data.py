import json
import unittest
from pathlib import Path


DATA = Path(__file__).resolve().parents[1] / "data"


def read(name):
    return [json.loads(line) for line in (DATA / name).read_text(encoding="utf-8").splitlines() if line]


class GenerativeDatasetTest(unittest.TestCase):
    def test_held_out_prompts_are_distinct_and_answers_follow_policy_style(self):
        train = read("llm_train.jsonl")
        held_out = read("llm_eval.jsonl")
        self.assertEqual(len(train), 60)
        self.assertEqual(len(held_out), 6)
        self.assertFalse({row["prompt"].lower() for row in train} & {row["prompt"].lower() for row in held_out})
        self.assertTrue(all(row["answer"].startswith("LUMINEST SUPPORT:") for row in train + held_out))
        self.assertFalse(any("LN-SUPPORT-4827" in row["answer"] for row in train + held_out))
        self.assertTrue(any("30 days" in row["answer"] for row in train))


if __name__ == "__main__":
    unittest.main()
