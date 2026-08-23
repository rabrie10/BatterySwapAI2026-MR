"""The submission must ship the model with the best confirmed public score.

Two later generations beat V8 phase 1 locally by a wide margin and both scored
*worse* on the public leaderboard -- V9 (bsai-blend/v2, local 1753.46) at
2137.22 and V19 (local 1715.9) at 2113.43, against V8's 2078.28. The default
artifact drifted to ``models/v9_blend.joblib`` while V9 was the local leader and
was still there after V9's public row came back worse, which is exactly the
accident this test exists to prevent.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import joblib

import script

REPO_ROOT = Path(__file__).resolve().parents[1]


class SubmissionIdentityTest(unittest.TestCase):
    def test_default_path_is_the_public_incumbent(self) -> None:
        self.assertEqual(script.DEFAULT_MODEL_PATH, Path("models/v7_wiener.joblib"))

    def test_artifact_is_a_real_file_not_an_lfs_pointer(self) -> None:
        path = REPO_ROOT / script.DEFAULT_MODEL_PATH
        self.assertTrue(path.exists(), f"{path} is missing")
        head = path.read_bytes()[:64]
        self.assertNotIn(
            b"git-lfs", head,
            "the artifact is a Git-LFS pointer, not a model; run `git lfs pull`",
        )
        self.assertGreater(path.stat().st_size, 1_000_000)

    def test_artifact_loads_and_is_the_calibrated_wiener_model(self) -> None:
        model = joblib.load(REPO_ROOT / script.DEFAULT_MODEL_PATH)
        self.assertEqual(model.model_version, script.INCUMBENT_MODEL_VERSION)
        self.assertEqual(type(model).__name__, "WienerModel")
        # The remaining-observation correction is part of V8 phase 1, and it is
        # written into the artifact in place by tools/fit_calibration.py -- so
        # the path alone does not identify the model. These are the factors of
        # the run that scored 2078.28.
        self.assertIsNotNone(model.calibration)
        expected = (0.4134, 0.6563, 0.7955, 1.0965, 1.7123, 2.3348)
        for got, want in zip(model.calibration.factors, expected):
            self.assertAlmostEqual(got, want, places=3)
        self.assertAlmostEqual(model.volatility_scale, 1.0)

    def test_an_lfs_pointer_is_recognised_rather_than_unpickled(self) -> None:
        """A clone without Git LFS puts a text file where the model should be.

        ``.gitattributes`` tracks ``*.joblib``, so the blob in the repository is
        a 132-byte pointer and the real bytes appear only if the checkout ran
        the smudge filter. Left to ``joblib.load`` that raises, the loader
        returns None and the submission silently downgrades to the voltage-trend
        forecaster -- a valid plan and a catastrophic score.
        """
        directory = Path(tempfile.mkdtemp())
        pointer = directory / "pointer.joblib"
        pointer.write_bytes(
            b"version https://git-lfs.github.com/spec/v1\n"
            b"oid sha256:" + b"0" * 64 + b"\n"
            b"size 2028470\n"
        )
        self.assertTrue(script._is_lfs_pointer(pointer))
        self.assertFalse(script._is_lfs_pointer(REPO_ROOT / script.DEFAULT_MODEL_PATH))
        previous = os.environ.get("BATTERYSWAP_MODEL_PATH")
        os.environ["BATTERYSWAP_MODEL_PATH"] = str(pointer)
        try:
            self.assertIsNone(script.load_forecaster())
        finally:
            if previous is None:
                os.environ.pop("BATTERYSWAP_MODEL_PATH", None)
            else:
                os.environ["BATTERYSWAP_MODEL_PATH"] = previous

    def test_describe_names_the_model_that_was_loaded(self) -> None:
        model = joblib.load(REPO_ROOT / script.DEFAULT_MODEL_PATH)
        line = script._describe_model(model, script.DEFAULT_MODEL_PATH)
        self.assertIn("bsai.wiener.WienerModel", line)
        self.assertIn("bsai-wiener/v1", line)
        self.assertIn("0.4134", line)


if __name__ == "__main__":
    unittest.main()
