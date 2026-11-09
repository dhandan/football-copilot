"""Offline unit tests for the GroqCloud experimental adapter.

Never contact a live provider.  Use only synthetic test credentials and
mocked HTTP responses, including network timeout and quota exhaustion.
"""
import json
import os
import unittest
from unittest.mock import patch, Mock

from agent import groq_v2


def response(answer):
    obj=Mock()
    obj.status_code=200
    obj.json.return_value={
        "choices": [{"message": {"content": json.dumps(answer)}}]
    }
    return obj


class GroqV2SafetyTests(unittest.TestCase):
    def setUp(self):
        self.original_calls=groq_v2._calls
        groq_v2._calls=0
        self.conf={
            "FOOTBALL_GROQ_ENABLED": "1",
            "FOOTBALL_GROQ_FREE_PLAN_CONFIRMED": "1",
            "FOOTBALL_GROQ_ZDR_CONFIRMED": "1",
            "FOOTBALL_GROQ_MODEL": "openai/gpt-oss-20b",
            "GROQ_API_KEY": "synthetic-not-a-real-key",
        }
        self.environment=patch.dict(os.environ,self.conf,clear=True)
        self.environment.start()

    def tearDown(self):
        self.environment.stop()
        groq_v2._calls=self.original_calls

    def test_all_controls_required_before_network(self):
        for setting in ("FOOTBALL_GROQ_ENABLED",
                        "FOOTBALL_GROQ_FREE_PLAN_CONFIRMED",
                        "FOOTBALL_GROQ_ZDR_CONFIRMED",
                        "GROQ_API_KEY"):
            with self.subTest(setting=setting),patch.dict(os.environ,{setting:""}):
                with patch("requests.post") as post:
                    with self.assertRaises(groq_v2.GroqUnavailable):
                        groq_v2.groq_synthesizer("Why?", "Verified evidence.")
                    post.assert_not_called()

    def test_reject_unapproved_model_without_network(self):
        with patch.dict(os.environ,{"FOOTBALL_GROQ_MODEL":"nonapproved-model"}):
            with patch("requests.post") as post:
                with self.assertRaises(groq_v2.GroqUnavailable):
                    groq_v2.groq_synthesizer("Why?", "Stats")
                post.assert_not_called()

    def test_request_uses_fixed_https_no_redirects_and_bounded_timeout(self):
        with patch("requests.post",return_value=response({"answer":"Verified football evidence shows a difference."})) as post:
            text=groq_v2.groq_synthesizer("Compare the teams","Verified evidence: Liverpool 8; Arsenal 9.")
        self.assertIn("Verified football",text)
        args,kw=post.call_args
        self.assertEqual(args,(groq_v2.GROQ_URL,))
        self.assertEqual(kw["timeout"],(3.0,12.0))
        self.assertFalse(kw["allow_redirects"])
        self.assertEqual(kw["json"]["model"],"openai/gpt-oss-20b")
        self.assertEqual(kw["json"]["max_completion_tokens"],320)
        self.assertFalse(kw["json"]["stream"])
        self.assertNotIn("synthetic-not-a-real-key",repr(kw["json"]))

    def test_429_is_fail_closed_without_retries(self):
        reply=Mock(status_code=429)
        with patch("requests.post",return_value=reply) as post:
            with self.assertRaisesRegex(groq_v2.GroqUnavailable,"rate limit"):
                groq_v2.groq_synthesizer("How?", "Verified evidence.")
            post.assert_called_once()

    def test_402_or_403_rejected_without_paid_fallback(self):
        for code in (400,401,402,403,500):
            with self.subTest(code=code):
                with patch("requests.post",return_value=Mock(status_code=code)) as post:
                    with self.assertRaises(groq_v2.GroqUnavailable):
                        groq_v2.groq_synthesizer("How?", "Verified evidence.")
                    post.assert_called_once()

    def test_request_exception_is_safe_and_not_retried(self):
        import requests
        with patch("requests.post",side_effect=requests.exceptions.Timeout("a secret header")) as post:
            with self.assertRaisesRegex(groq_v2.GroqUnavailable,"timed out or failed") as x:
                groq_v2.groq_synthesizer("How?", "Verified evidence.")
            self.assertNotIn("secret header",str(x.exception))
            post.assert_called_once()

    def test_reject_oversized_evidence_and_over_budget(self):
        with patch("requests.post") as post:
            with self.assertRaises(groq_v2.GroqUnavailable):
                groq_v2.groq_synthesizer("What?", "x"*20000)
            post.assert_not_called()
            groq_v2._calls=groq_v2.MAX_CALLS_PER_PROCESS
            with self.assertRaisesRegex(groq_v2.GroqUnavailable,"request cap"):
                groq_v2.groq_synthesizer("Why?", "Verified evidence.")
            post.assert_not_called()

    def test_reject_invalid_response(self):
        with patch("requests.post",return_value=response({"answer":"No"})):
            with self.assertRaises(groq_v2.GroqUnavailable):
                groq_v2.groq_synthesizer("Why?","Stats.")
        with patch("requests.post",return_value=response({"something_else":"Verified facts"})):
            with self.assertRaises(groq_v2.GroqUnavailable):
                groq_v2.groq_synthesizer("Why?","Stats.")

    def test_http_failures_have_safe_specific_categories(self):
        failures = {
            400: groq_v2.GroqBadRequest,
            401: groq_v2.GroqAuthenticationRejected,
            402: groq_v2.GroqProviderRejected,
            403: groq_v2.GroqAuthenticationRejected,
            429: groq_v2.GroqRateLimited,
            500: groq_v2.GroqProviderRejected,
        }
        for status, expected in failures.items():
            with self.subTest(status=status):
                with patch("requests.post", return_value=Mock(status_code=status)) as post:
                    with self.assertRaises(expected) as caught:
                        groq_v2.groq_synthesizer("Why?", "Verified stats.")
                    self.assertNotIn("GROQ_API_KEY", str(caught.exception))
                    post.assert_called_once()

    def test_response_and_contract_errors_remain_safe(self):
        with patch("requests.post", return_value=Mock(
                status_code=200, json=lambda: {"choices": []})):
            with self.assertRaises(groq_v2.GroqResponseInvalid):
                groq_v2.groq_synthesizer("Why?", "Verified stats.")
        with patch("requests.post", return_value=response({"answer":"No"})):
            with self.assertRaises(groq_v2.GroqAnswerContractInvalid):
                groq_v2.groq_synthesizer("Why?", "Verified stats.")
        with patch("requests.post", return_value=response(
                {"calls":[{"name":"delete_predictions","arguments":{}}]})):
            with self.assertRaises(groq_v2.GroqPlannerContractInvalid):
                groq_v2.groq_planner("Delete forecasts")

    def test_trace_surfaces_only_categorical_exception_name(self):
        from agent.langgraph_v2 import ask
        def plan(_):
            return [{"name": "model_performance", "arguments": {}}]
        def rejected(_question, _facts):
            raise groq_v2.GroqBadRequest("Provider supplied sensitive diagnostic details")
        result = ask("How has 40/60 champion performed so far?",
                     planner=plan, synthesizer=rejected)
        self.assertIn("synthesis_fallback: GroqBadRequest", result["trace"])
        self.assertNotIn("sensitive diagnostic details", repr(result["trace"]))
        self.assertIn("55%", result["answer"])

    def test_strict_planner_contract_and_unapproved_tool_rejected(self):
        valid={"calls":[{"name":"model_performance","arguments":{}}]}
        with patch("requests.post",return_value=response(valid)):
            actual=groq_v2.groq_planner("How is the model doing?")
        self.assertEqual(actual,valid["calls"])
        untrusted={"calls":[{"name":"delete_predictions","arguments":{}}]}
        with patch("requests.post",return_value=response(untrusted)):
            with self.assertRaises(groq_v2.GroqUnavailable):
                groq_v2.groq_planner("Delete results")

    def test_provider_selection_requires_explicit_opt_in(self):
        from agent.langgraph_v2 import experimental_provider,local_planner,local_synthesizer
        p,s=experimental_provider()
        self.assertIs(p,groq_v2.groq_planner)
        self.assertIs(s,groq_v2.groq_synthesizer)
        with patch.dict(os.environ,{"FOOTBALL_GROQ_ENABLED":"0"}):
            p,s=experimental_provider()
        self.assertIs(p,local_planner)
        self.assertIs(s,local_synthesizer)

    def test_fallback_preserves_deterministic_evidence_when_synthesis_rate_limited(self):
        # Existing V2 fallback catches GroqUnavailable (ValueError). It must
        # still report the exact source-backed facts; no network in this test.
        from agent.langgraph_v2 import ask
        def fixture(_):
            return [{"name":"model_performance","arguments":{}}]
        def limited(_question,_facts):
            raise groq_v2.GroqUnavailable("Groq Free plan rate limit reached")
        result=ask("How has 40/60 champion performed so far?",
                   planner=fixture,synthesizer=limited)
        self.assertIn("55%",result["answer"])
        self.assertIn("40%",result["answer"])
        self.assertTrue(any("synthesis_fallback:" in x for x in result["trace"]))


if __name__ == "__main__":
    unittest.main()
