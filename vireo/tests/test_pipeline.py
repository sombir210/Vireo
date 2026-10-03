import unittest

from pipeline import infer_reason_code_from_text, normalize_reason_code, parse_llm_json


class PipelineTests(unittest.TestCase):
    def test_normalize_reason_code_preserves_policy_format(self):
        self.assertEqual(normalize_reason_code("doa repl"), "DOA-REPL")
        self.assertEqual(normalize_reason_code("dup_payment"), "DUP-PAYMENT")
        self.assertEqual(normalize_reason_code("PRICE ADJ"), "PRICE-ADJ")

    def test_infer_reason_code_from_text_matches_policy_codes(self):
        self.assertEqual(infer_reason_code_from_text("customer says order was cancelled before dispatch"), "CANCEL")
        self.assertEqual(infer_reason_code_from_text("duplicate payment charged twice"), "DUP-PAYMENT")
        self.assertEqual(infer_reason_code_from_text("warranty buyback request"), "WTY-BUYBACK")

    def test_parse_llm_json_handles_policy_codes_and_amounts(self):
        payload = parse_llm_json('{"refund_issued": true, "reason_code": "GW-OTHER", "amount": 350.5}')
        self.assertEqual(payload, {"refund_issued": True, "reason_code": "GW-OTHER", "amount": 350.5})


if __name__ == "__main__":
    unittest.main()
