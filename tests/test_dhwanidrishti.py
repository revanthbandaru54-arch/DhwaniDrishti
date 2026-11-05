"""
Automated Unit & API Integration Tests for DhwaniDrishti
========================================================
Run via:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import asyncio
import unittest

from backend.grammar_engine import ISLGrammarEngine
from backend.dtw_evaluator import STEM_BENCHMARKS, evaluate_stem_sign
from backend.sequence_classifier import TemporalSignSequenceClassifier
from backend.main import (
    LectureCastParseRequest,
    DoubtDeskClassifyRequest,
    StemCoachEvaluateRequest,
    health_check,
    parse_lecture_speech,
    classify_student_doubt,
    get_stem_terms,
    evaluate_stem_practice,
)


class TestDhwaniDrishti(unittest.TestCase):
    def setUp(self) -> None:
        self.grammar = ISLGrammarEngine()
        self.classifier = TemporalSignSequenceClassifier()

    def test_lecturecast_svo_to_isl_sov(self) -> None:
        res = self.grammar.parse_to_isl(
            "Today the teacher explains the derivative algorithm to the students?",
            "en-IN",
        )
        self.assertIn("TODAY", res["isl_sov_gloss"])
        self.assertIn("IX-LEFT", res["isl_sov_gloss"])
        self.assertIn("IX-RIGHT", res["isl_sov_gloss"])
        # Verb "EXPLAIN" should appear after subject and object in SOV order
        subj_idx = res["isl_sov_gloss"].index("TEACHER")
        verb_idx = res["isl_sov_gloss"].index("EXPLAIN")
        self.assertLess(subj_idx, verb_idx)
        self.assertTrue(len(res["avatar_keyframes"]) >= 4)

    def test_multilingual_telugu_and_hindi_normalization(self) -> None:
        te_res = self.grammar.parse_to_isl("ఈరోజు ఉపాధ్యాయుడు విద్యార్థులు అల్గారిథమ్ వివరిస్తారు?", "te-IN")
        self.assertIn("ALGORITHM", te_res["isl_sov_gloss"])
        hi_res = self.grammar.parse_to_isl("आज शिक्षक छात्र अवकलन समीकरण समझाते?", "hi-IN")
        self.assertIn("DERIVATIVE", hi_res["isl_sov_gloss"])

    def test_stem_coach_dtw_and_cosine_evaluation(self) -> None:
        ref_traj = STEM_BENCHMARKS["Algorithm"]["trajectory"].tolist()
        # 1. Exact benchmark alignment should pass with 100% score and all green joints
        perfect = evaluate_stem_sign(
            "Algorithm",
            ref_traj,
            measured_palm_angle_deg=25.0,
            measured_eyebrow_raise=0.35,
        )
        self.assertEqual(perfect["dtw_distance"], 0.0)
        self.assertAlmostEqual(perfect["cosine_similarity"], 1.0, places=4)
        self.assertTrue(perfect["passed"])
        self.assertEqual(perfect["joint_status"]["right_wrist"], "green")

        # 2. Deviated palm angle should flag red wrist/index joints and emit degree hint
        deviated = evaluate_stem_sign(
            "Algorithm",
            ref_traj,
            measured_palm_angle_deg=1.0,
            measured_eyebrow_raise=0.05,
        )
        self.assertEqual(deviated["joint_status"]["right_wrist"], "red")
        self.assertEqual(deviated["joint_status"]["eyebrows"], "red")
        self.assertTrue(any("Angle your right palm" in h for h in deviated["actionable_hints"]))

    def test_doubtdesk_sequence_classifier(self) -> None:
        ref_traj = STEM_BENCHMARKS["Derivative"]["trajectory"].tolist()
        out = self.classifier.classify_sequence(ref_traj, target_language="te-IN", hint_index=0)
        self.assertTrue(out["is_dynamic_sequence"])
        self.assertEqual(out["frames_analyzed"], 30)
        self.assertIn("en", out["translations"])
        self.assertIn("te", out["translations"])
        self.assertIn("hi", out["translations"])

    @staticmethod
    def _run_coro(coro):
        try:
            coro.send(None)
        except StopIteration as stop:
            return stop.value
        finally:
            coro.close()

    def test_fastapi_async_endpoints(self) -> None:
        health = self._run_coro(health_check())
        self.assertEqual(health["status"], "ok")
        self.assertEqual(health["holistic_landmarks"], 543)

        terms = self._run_coro(get_stem_terms())
        self.assertEqual(len(terms["terms"]), 5)

        parsed = self._run_coro(
            parse_lecture_speech(LectureCastParseRequest(text="Students learn photosynthesis", language="en-IN"))
        )
        self.assertIn("PHOTOSYNTHESIS", parsed["isl_sov_gloss"])


if __name__ == "__main__":
    unittest.main()

