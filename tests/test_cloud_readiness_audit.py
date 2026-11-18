"""Unit tests for the redacted detector (no actual credentials)."""
import unittest
from scripts.audit_cloud_readiness import classify


class AuditDetectorTests(unittest.TestCase):
    def test_placeholder_is_not_a_secret(self):
        self.assertEqual(classify("ODDS_API_KEY=your_odds_api_key_here"), [])
        self.assertEqual(classify("GROQ_API_KEY=example"), [])

    def test_suspicious_assignment_never_echoes_value(self):
        placeholder_test_value = "not-a-real-token-123456789"
        results = classify("ODDS_API_KEY=" + placeholder_test_value)
        self.assertEqual(results, ["api-token-assignment"])
        self.assertNotIn(placeholder_test_value, repr(results))

    def test_private_key_header(self):
        self.assertEqual(classify("-----BEGIN PRIVATE KEY-----"), ["private-key-header"])


if __name__ == "__main__":
    unittest.main()
