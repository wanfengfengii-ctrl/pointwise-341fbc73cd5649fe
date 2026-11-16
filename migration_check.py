"""Public business probes; no migration or reward-to-matrix implementation."""
import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

from analysis_archive import Archive, ResultNotReady, ValidationError
from analysis_archive.samples import build_legacy, counter, stopped
from vendor import sdk1, sdk2


def close(actual, expected):
    if isinstance(expected, dict):
        assert set(actual) == set(expected), (actual, expected)
        for key in expected:
            close(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected), (actual, expected)
        for a, b in zip(actual, expected):
            close(a, b)
    elif isinstance(expected, (int, float)):
        assert math.isfinite(actual)
        assert abs(actual - expected) <= 2e-8 + 2e-8 * abs(expected), (actual, expected)
    else:
        assert actual == expected, (actual, expected)


def native_counter():
    return {
        "state_ids": ["running"], "metric_ids": ["cycles", "load"],
        "residence": [[0, 0]],
        "events": [{"id": "native-pulse", "source": 0, "target": 0,
                    "rate": 1, "payment": [1, -1]}],
    }


def native_stopped():
    return {
        "state_ids": ["on", "off"], "metric_ids": ["uptime", "stops"],
        "residence": [[1, 0], [0, 0]],
        "events": [{"id": "stop", "source": 0, "target": 1,
                    "rate": 3, "payment": [0, 1]}],
    }


def stopped_expected(rate, time):
    survival = math.exp(-rate * time)
    probability = 1 - survival
    et = probability / rate
    et2 = 2 * (1 - survival * (1 + rate * time)) / rate ** 2
    etn = (1 - survival * (1 + rate * time)) / rate
    return {
        "metrics": ["uptime", "stops"], "states": ["on", "off"],
        "probabilities": [survival, probability], "means": [et, probability],
        "covariance": [[et2 - et * et, etn - et * probability],
                       [etn - et * probability, probability * (1 - probability)]],
    }


def legacy_probe():
    with Archive() as archive:
        archive.put_model("counter", counter())
        archive.put_cohort("c", ["running"], {"running": 1}, 1)
        result = archive.run_task(archive.create_task("counter", "counter", "c"))
        close(result["statistics"], {
            "metrics": ["cycles", "load"], "states": ["running"],
            "probabilities": [1], "means": [2, 4],
            "covariance": [[2, 4], [4, 8]],
        })
        archive.put_model("stop", stopped())
        archive.put_cohort("s", ["on", "off"], {"on": 1}, 1)
        other = archive.run_task(archive.create_task("stop", "stop", "s"))
        close(other["statistics"], stopped_expected(2, 1))
        return {"self_event": result["statistics"], "stopping": other["statistics"]}


class InjectedFailure(RuntimeError):
    pass


def fail_at(target):
    def fault(stage):
        if stage == target:
            raise InjectedFailure(stage)
    return fault


def dump(db):
    return tuple(db.iterdump())


def format_number(archive):
    return archive.store.connection.execute(
        "SELECT value FROM archive_meta WHERE key='format'").fetchone()[0]


@contextmanager
def target_only(archive):
    db = archive.store.connection
    calls = [0]
    previous = sys.getprofile()

    def authorizer(action, table, column, database, trigger):
        if table in {"model_revisions", "model_heads"} and action in {
                sqlite3.SQLITE_READ, sqlite3.SQLITE_INSERT,
                sqlite3.SQLITE_UPDATE, sqlite3.SQLITE_DELETE}:
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    def profile(frame, event, arg):
        if event == "call":
            if frame.f_code is sdk1.analyze.__code__:
                raise AssertionError("SDK1 used after migration")
            if frame.f_code is sdk2.propagate.__code__:
                calls[0] += 1

    db.set_authorizer(authorizer)
    sys.setprofile(profile)
    try:
        yield calls
    finally:
        sys.setprofile(previous)
        db.set_authorizer(None)


def scenario():
    from analysis_archive.migration import upgrade

    with tempfile.TemporaryDirectory(prefix="reward-migration-") as directory:
        path = Path(directory) / "archive.sqlite"
        ids = build_legacy(path)
        with Archive(path) as archive:
            assert format_number(archive) == "1"
            original = archive.get_result(ids["completed"])
            before = dump(archive.store.connection)

        for stage in ("after_models", "before_commit"):
            with Archive(path) as archive:
                try:
                    upgrade(archive, fault=fail_at(stage))
                except InjectedFailure as exc:
                    assert str(exc) == stage
                else:
                    raise AssertionError("migration fault boundary not reached")
                assert dump(archive.store.connection) == before
                assert format_number(archive) == "1"
                assert archive.get_result(ids["completed"]) == original
            with Archive(path) as archive:
                assert dump(archive.store.connection) == before

        with Archive(path) as archive:
            retained_models, retained_analysis = archive.models, archive.analysis
            upgrade(archive)
            assert format_number(archive) == "2"
            assert archive.get_result(ids["completed"]) == original
            assert archive.get_task(ids["pending"])["revision"] == 1

            with target_only(archive) as calls:
                assert retained_models.revisions("press") == [1, 2]
                historical = archive.run_task(ids["pending"])
                close(historical["statistics"], original["statistics"])
                close(retained_analysis.state_probabilities("stopping", {"on": 1}, 1),
                      {"on": math.exp(-2), "off": 1 - math.exp(-2)})

                assert archive.put_native_model("press", native_counter()) == 3
                assert retained_models.get_model("press")["revision"] == 3
                definition = archive.get_model("press")["definition"]
                assert definition["events"][0]["id"] == "native-pulse"
                assert definition["events"][0]["source"] == "running"
                assert definition["events"][0]["reward"] == ["1", "-1"]
                task = archive.create_task("native-run", "press", "shift")
                native_report = archive.run_task(task)
                close(native_report["statistics"], {
                    "metrics": ["cycles", "load"], "states": ["running"],
                    "probabilities": [1], "means": [2, -2],
                    "covariance": [[2, -2], [-2, 2]],
                })
                assert archive.run_task(ids["pending"]) == historical
                assert archive.put_native_model("stopping", native_stopped()) == 2
                close(retained_analysis.state_probabilities("stopping", {"on": 1}, 1),
                      {"on": math.exp(-3), "off": 1 - math.exp(-3)})
                stop_task = archive.create_task("native-stop", "stopping", "start-on")
                stop_report = archive.run_task(stop_task)
                close(stop_report["statistics"], stopped_expected(3, 1))

                bad = native_counter()
                bad["events"][0]["source"] = 1
                try:
                    archive.put_native_model("press", bad)
                except ValidationError:
                    pass
                else:
                    raise AssertionError("unknown native state index accepted")
                assert archive.revisions("press") == [1, 2, 3]

                assert archive.put_model("press", counter(4)) == 4
                retry_task = archive.create_task("old-write-new-storage", "press", "shift")
                for stage in ("after_compute", "after_result_insert", "before_commit"):
                    try:
                        archive.run_task(retry_task, fault=fail_at(stage))
                    except InjectedFailure:
                        pass
                    else:
                        raise AssertionError("result fault boundary not reached")
                    assert archive.get_task(retry_task)["status"] == "pending"
                    try:
                        archive.get_result(retry_task)
                    except ResultNotReady:
                        pass
                    else:
                        raise AssertionError("partial result became visible")
                assert calls[0] > 0

        with Archive(path) as archive:
            before_noop = dump(archive.store.connection)
            upgrade(archive, fault=lambda stage: (_ for _ in ()).throw(
                AssertionError("idempotent upgrade invoked fault")))
            assert dump(archive.store.connection) == before_noop
            with target_only(archive) as calls:
                assert archive.get_result(ids["completed"]) == original
                assert archive.get_result(task) == native_report
                assert archive.get_result(stop_task) == stop_report
                assert archive.run_task(ids["pending"]) == historical
                retried = archive.run_task(retry_task)
                close(retried["statistics"], {
                    "metrics": ["cycles", "load"], "states": ["running"],
                    "probabilities": [1], "means": [8, 16],
                    "covariance": [[8, 16], [16, 32]],
                })
                assert archive.run_task(retry_task) == retried
                assert calls[0] > 0
            assert archive.store.connection.execute(
                "PRAGMA integrity_check").fetchone()[0] == "ok"
            assert archive.store.connection.execute(
                "PRAGMA foreign_key_check").fetchall() == []
            return {"historical": historical, "native_self": native_report,
                    "native_stopping": stop_report, "reopened_retry": retried,
                    "format": format_number(archive)}


def migrated_reward_suite():
    from analysis_archive.migration import upgrade
    from tests.test_reward_semantics import RewardSemantics

    class MigratedRewards(RewardSemantics):
        def evaluate(self, definition, initial, time):
            with Archive() as archive:
                upgrade(archive)
                archive.put_model("m", definition)
                archive.put_cohort("c", [s["id"] for s in definition["states"]], initial, time)
                return archive.run_task(archive.create_task("r", "m", "c"))["statistics"]

    return unittest.defaultTestLoader.loadTestsFromTestCase(MigratedRewards)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("legacy", "verify", "demo"))
    mode = parser.parse_args().mode
    if mode == "legacy":
        output = legacy_probe()
    else:
        if mode == "verify":
            suite = unittest.defaultTestLoader.discover("tests")
            suite.addTests(migrated_reward_suite())
            if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
                raise SystemExit(1)
        output = scenario()
    print(json.dumps(output, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
