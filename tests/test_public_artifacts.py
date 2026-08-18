from __future__ import annotations

import csv
import hashlib
import json
import unittest
from pathlib import Path

from project_config import MODEL_REVISIONS


REPO_ROOT = Path(__file__).resolve().parents[1]


class PublicArtifactTests(unittest.TestCase):
    def test_results_manifest_matches_public_artifacts(self):
        manifest = json.loads(
            (REPO_ROOT / "results/manifest.json").read_text(encoding="utf-8")
        )
        artifacts = manifest["artifacts"]
        self.assertEqual(manifest["artifact_count"], len(artifacts))
        for artifact in artifacts:
            path = REPO_ROOT / artifact["path"]
            self.assertTrue(path.is_file(), artifact["path"])
            self.assertEqual(path.stat().st_size, artifact["bytes"])
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(),
                artifact["sha256"],
                artifact["path"],
            )

    def test_public_dataset_is_only_a_small_schema_sample(self):
        with (REPO_ROOT / "outcome_3cls.csv").open(
            newline="", encoding="utf-8-sig"
        ) as handle:
            rows = list(csv.reader(handle))
        self.assertLessEqual(len(rows) - 1, 500)
        self.assertTrue(
            {"CDSR.id", "outcome.id", "outcome", "outcome.class"}.issubset(
                set(rows[0])
            )
        )

    def test_model_revision_manifest_matches_python_config(self):
        manifest = json.loads(
            (REPO_ROOT / "config/model_revisions.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest, MODEL_REVISIONS)

    def test_subgroup_mapping_is_complete(self):
        payload = json.loads(
            (REPO_ROOT / "config/subgroup_mapping.json").read_text(encoding="utf-8")
        )
        rows = payload["subgroups"]
        self.assertEqual([row["code"] for row in rows], list(range(1, 17)))
        self.assertEqual(rows[0]["mapped_label_id"], 0)
        self.assertTrue(
            all(row["mapped_label_id"] == 1 for row in rows[1:7])
        )
        self.assertTrue(
            all(row["mapped_label_id"] == 2 for row in rows[7:])
        )

    def test_reviewer_agreement_matches_manuscript(self):
        result = json.loads(
            (
                REPO_ROOT
                / "results/analyses/reviewer_agreement/reviewer_agreement.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(result["double_coded_subset_n"], 948)
        self.assertAlmostEqual(result["subgroup_kappa"], 0.9338108507415235)
        self.assertAlmostEqual(
            result["mapped_three_class_kappa"], 0.9417449569255858
        )

    def test_primary_checkpoint_metrics_match_full_test_set(self):
        result = json.loads(
            (REPO_ROOT / "results/primary_checkpoint_metrics.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(result["n_test_samples"], 4503)
        self.assertAlmostEqual(result["accuracy"], 4122 / 4503)
        self.assertAlmostEqual(result["macro_f1"], 0.9151846159003062)
        self.assertAlmostEqual(result["time_per_sample"], 0.0019931660846368806)
        self.assertIn("validation accuracy", result["paper_role"])
        self.assertIn("tie-breaker", result["provenance_note"])

    def test_core_evaluation_excludes_obsolete_hybrid_and_item_bootstrap(self):
        results = json.loads(
            (REPO_ROOT / "results/evaluation_results.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(len(results), 2)
        self.assertFalse(
            any("hybrid" in row["method"].lower() for row in results)
        )
        for row in results:
            self.assertNotIn("accuracy_ci", row)
            self.assertNotIn("accuracy_ci_95", row)
            self.assertNotIn("macro_f1_ci", row)
            self.assertNotIn("macro_f1_ci_95", row)

        source = (REPO_ROOT / "evaluate_all_methods.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("OUTCOME_RUN_LEGACY_HYBRID", source)
        self.assertNotIn("LEGACY_CONFIDENCE_THRESHOLD", source)
        self.assertNotIn("def bootstrap_ci", source)

    def test_search_selection_provenance_matches_manuscript(self):
        rows = json.loads(
            (REPO_ROOT / "results/search_results.json").read_text(
                encoding="utf-8"
            )
        )
        primary = [row for row in rows if row.get("is_primary_reference")]
        self.assertEqual(len(primary), 1)
        self.assertIn("validation accuracy", primary[0]["provenance_note"])
        self.assertIn("tie-breaker", primary[0]["provenance_note"])

    def test_current_uncertainty_and_ensemble_artifacts_are_unambiguous(self):
        uncertainty_path = (
            REPO_ROOT / "results/analyses/uncertainty/table2_uncertainty.csv"
        )
        with uncertainty_path.open(newline="", encoding="utf-8") as handle:
            rows = {row["model"]: row for row in csv.DictReader(handle)}
        bioformer = rows["Bioformer-8L"]
        self.assertAlmostEqual(
            float(bioformer["accuracy_ci_lower"]), 0.9058345720444235
        )
        self.assertAlmostEqual(
            float(bioformer["accuracy_ci_upper"]), 0.924868335725735
        )

        self.assertFalse((REPO_ROOT / "results/ensemble_results.json").exists())
        ensemble = json.loads(
            (
                REPO_ROOT / "results/ensemble_results_aligned_primary.json"
            ).read_text(encoding="utf-8")
        )
        self.assertAlmostEqual(
            ensemble["ensemble"]["accuracy"], 0.9111703308905175
        )
        self.assertAlmostEqual(
            ensemble["ensemble"]["macro_f1"], 0.9176184142719231
        )

    def test_subgroup_totals_match_primary_checkpoint(self):
        result = json.loads(
            (
                REPO_ROOT
                / "results/analyses/subgroup_performance/subgroup_performance.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(result["test_outcomes_n"], 4503)
        self.assertEqual(result["correct_n"], 4122)
        self.assertEqual(
            result["prediction_counts"],
            {"Objective": 340, "Semi-objective": 1973, "Subjective": 2190},
        )

    def test_locked_hybrid_threshold_is_current(self):
        result = json.loads(
            (
                REPO_ROOT
                / "results/analyses/hybrid_threshold/locked_test_results.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(result["selected_threshold"], 0.55)

    def test_gpt_cluster_comparison_matches_reported_difference(self):
        result = json.loads(
            (
                REPO_ROOT
                / "results/analyses/gpt_prompt_sensitivity/"
                "gpt_prompt_sensitivity_cluster_analysis.json"
            ).read_text(encoding="utf-8")
        )
        comparison = result["differences"]["18_examples_minus_10_examples"]
        self.assertAlmostEqual(
            comparison["accuracy"]["point_difference"], 7 / 4503
        )
        self.assertAlmostEqual(
            result["mcnemar_10_vs_18"]["exact_two_sided_p"],
            0.608384106899959,
        )


if __name__ == "__main__":
    unittest.main()
