"""Independent business checks for the format-two target representation.

These assertions use closed-form business formulas only (Poisson counts,
mixtures, absorbing chains).  They do not call or mirror SDK1.
"""
import math
import tempfile
import unittest
from pathlib import Path

from analysis_archive import Archive, ArchiveError, ValidationError
from analysis_archive.migration import upgrade
from analysis_archive.samples import counter


def close(test, actual, expected):
    for a, b in zip(actual, expected):
        test.assertTrue(math.isclose(a, b, abs_tol=2e-8, rel_tol=2e-8), (a, b))


class TargetBusiness(unittest.TestCase):
    def evaluate(self, definition, initial, duration, revision=None):
        with Archive() as archive:
            upgrade(archive)
            archive.put_model("m", definition)
            states = [s["id"] for s in definition["states"]]
            archive.put_cohort("c", states, initial, duration)
            return archive.run_task(archive.create_task("r", "m", "c"))["statistics"]

    def test_residence_plus_poisson_self_event(self):
        # Deterministic residence r*t plus Poisson(rate*t) event counts.
        definition = counter(2, (1, 2), ("1/2", -1))
        result = self.evaluate(definition, {"running": 1}, 1)
        close(self, result["means"], [0.5 + 2, -1 + 4])
        close(self, result["covariance"][0], [2, 4])
        close(self, result["covariance"][1], [4, 8])
        self.assertEqual(result["probabilities"], [1.0])

    def test_parallel_events_are_distinct_poisson_causes(self):
        # Two parallel self events keep the state fixed, so their counts are
        # independent Poisson variables.  Covariance therefore adds one rank-one
        # term per cause, and each event's vector payment stays joint.
        definition = {"metrics": ["a", "b"],
                      "states": [{"id": "s", "reward": ["1/2", -1]}],
                      "events": [{"id": "one", "source": "s", "target": "s",
                                  "rate": 2, "reward": [1, -1]},
                                 {"id": "two", "source": "s", "target": "s",
                                  "rate": "5/2", "reward": [3, 1]}]}
        time = "3/2"
        result = self.evaluate(definition, {"s": 1}, time)
        t = 1.5
        means = [0.5 * t + 2 * 1 * t + 2.5 * 3 * t, -1 * t + 2 * -1 * t + 2.5 * 1 * t]
        close(self, result["means"], means)
        raw = [[2 * 1 * 1 + 2.5 * 3 * 3, 2 * 1 * -1 + 2.5 * 3 * 1],
               [2 * 1 * -1 + 2.5 * 3 * 1, 2 * -1 * -1 + 2.5 * 1 * 1]]
        covariance = [[raw[i][j] * t for j in range(2)] for i in range(2)]
        for i in range(2):
            close(self, result["covariance"][i], covariance[i])
        close(self, result["probabilities"], [1])

    def test_same_occurrence_payment_is_not_independent(self):
        # One event paying (1,-1) gives perfectly anticorrelated metric counts.
        definition = counter(1, (1, -1))
        result = self.evaluate(definition, {"running": 1}, 2)
        close(self, result["means"], [2, -2])
        close(self, result["covariance"][0], [2, -2])
        close(self, result["covariance"][1], [-2, 2])

    def test_parallel_opposing_payments_keep_full_covariance(self):
        # Two causes with opposing second components cancel their mean exactly;
        # their variances must still add, not collapse to the mean residual.
        definition = {"metrics": ["a", "b"],
                      "states": [{"id": "s", "reward": [0, 0]}],
                      "events": [{"id": "up", "source": "s", "target": "s",
                                  "rate": 4, "reward": [8, 8]},
                                 {"id": "down", "source": "s", "target": "s",
                                  "rate": 4, "reward": [8, -8]}]}
        result = self.evaluate(definition, {"s": 1}, 8)
        close(self, result["means"], [512, 0])
        close(self, result["covariance"][0], [4096, 0])
        close(self, result["covariance"][1], [0, 4096])

    def test_random_initial_state_variance(self):
        definition = {"metrics": ["x"],
                      "states": [{"id": "lo", "reward": [1]}, {"id": "hi", "reward": [3]}],
                      "events": []}
        result = self.evaluate(definition, {"lo": "1/2", "hi": "1/2"}, 2)
        close(self, result["means"], [4])
        close(self, result["covariance"][0], [4])
        close(self, result["probabilities"], [0.5, 0.5])

    def test_absorbing_state_keeps_residence_reward(self):
        definition = {"metrics": ["x"],
                      "states": [{"id": "hot", "reward": [2]}, {"id": "cold", "reward": [5]}],
                      "events": [{"id": "cool", "source": "hot", "target": "cold",
                                  "rate": 1, "reward": [0]}]}
        result = self.evaluate(definition, {"cold": 1}, 4)
        close(self, result["means"], [20])
        close(self, result["covariance"][0], [0])
        close(self, result["probabilities"], [0, 1])


class NativeAndTarget(unittest.TestCase):
    def test_native_rejected_before_upgrade(self):
        with Archive() as archive:
            with self.assertRaises(ArchiveError):
                archive.put_native_model("m", {"state_ids": ["s"]})
            self.assertEqual(archive.store.format(), "1")
            self.assertEqual(archive.store.connection.execute(
                "SELECT COUNT(*) FROM model_revisions").fetchone()[0], 0)

    def test_native_and_legacy_share_one_revision_sequence(self):
        native = {"state_ids": ["s"], "metric_ids": ["c", "l"],
                  "residence": [[0, 0]],
                  "events": [{"id": "p", "source": 0, "target": 0, "rate": 2,
                              "payment": [1, 2]}]}
        with Archive() as archive:
            upgrade(archive)
            self.assertEqual(archive.put_native_model("m", native), 1)
            self.assertEqual(archive.put_model("m", counter(2)), 2)
            self.assertEqual(archive.revisions("m"), [1, 2])
            self.assertEqual(archive.get_model("m", 1)["definition"]["events"][0]["id"], "p")
            archive.put_cohort("c", ["running"], {"running": 1}, 1)
            # counter's state id only matches revision 2's canonical directory.
            archive.put_cohort("s", ["s"], {"s": 1}, 1)
            first = archive.run_task(archive.create_task("a", "m", "s", revision=1))
            close(self, first["statistics"]["means"], [2, 4])
            close(self, first["statistics"]["covariance"][0], [2, 4])

    def test_native_endpoint_validation(self):
        native = {"state_ids": ["a", "b"], "metric_ids": ["x"],
                  "residence": [[0], [0]],
                  "events": [{"id": "e", "source": 0, "target": 1, "rate": 1, "payment": [1]}]}
        with Archive() as archive:
            upgrade(archive)
            for key, value in [("source", 2), ("target", -1), ("source", True),
                               ("source", False), ("source", 0.0)]:
                bad = {**native, "events": [dict(native["events"][0])]}
                bad["events"][0][key] = value
                with self.subTest(key=key, value=value):
                    with self.assertRaises(ValidationError):
                        archive.put_native_model("m", bad)
            self.assertEqual(archive.store.connection.execute(
                "SELECT COUNT(*) FROM target_model_revisions").fetchone()[0], 0)

    def test_target_storage_survives_audit_removal(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "a.sqlite"
            with Archive(path) as archive:
                archive.put_model("m", counter())
                archive.put_cohort("c", ["running"], {"running": 1}, 1)
                old = archive.run_task(archive.create_task("old", "m", "c"))
                upgrade(archive)
                archive.store.connection.execute("PRAGMA foreign_keys=OFF")
                for table in ("legacy_results_audit", "legacy_tasks_audit",
                              "model_heads", "model_revisions"):
                    archive.store.connection.execute("DROP TABLE " + table)
                archive.store.connection.execute("PRAGMA foreign_keys=ON")
                self.assertEqual(archive.get_result(1), old)
                self.assertEqual(archive.revisions("m"), [1])
                close(self, archive.state_probabilities("m", {"running": 1}, 1)
                      and [archive.state_probabilities("m", {"running": 1}, 1)["running"]], [1])
                archive.put_model("m", counter(3))
                newer = archive.run_task(archive.create_task("new", "m", "c"))
                close(self, newer["statistics"]["means"], [3, 6])
                self.assertEqual(archive.run_task(1), old)
                self.assertEqual(archive.store.connection.execute(
                    "PRAGMA foreign_key_check").fetchall(), [])


if __name__ == "__main__":
    unittest.main()
