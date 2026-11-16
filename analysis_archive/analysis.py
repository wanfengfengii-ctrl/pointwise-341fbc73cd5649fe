"""Reward analysis backend, dispatched on the archive's current format.

Format one retains the legacy SDK1 path.  Once an archive is at format two every
read, computation and publication goes through target storage and the untouched
SDK2 matrix port; the audit model tables are never consulted.
"""
import json

from . import sdk2_analysis
from . import validation as v
from .errors import ArchiveError
from vendor.sdk1 import analyze as analyze_v1


class Analysis:
    def __init__(self, store, models, tasks, results):
        self.store, self.models, self.tasks, self.results = store, models, tasks, results

    def _statistics(self, db, record, parameters):
        if self.store.target(db):
            representation = record["representation"]
            if representation is None:
                raise ArchiveError("format-two revision is missing its representation")
            definition = sdk2_analysis.from_representation(representation)
            return sdk2_analysis.analyze(definition, parameters["initial"], parameters["duration"])
        return analyze_v1(record["definition"], parameters["initial"], parameters["duration"])

    def state_probabilities(self, model_id, initial, duration, revision=None):
        with self.store.transaction() as db:
            _, record = self.models._resolve(db, model_id, revision)
            parameters = v.cohort([s["id"] for s in record["definition"]["states"]], initial, duration)
            statistics = self._statistics(db, record, parameters)
            return dict(zip(statistics["states"], statistics["probabilities"]))

    def run_task(self, task_id, *, fault=None):
        if fault is None:
            def fault(stage):
                return None
        with self.store.transaction() as db:
            task = self.tasks._row(db, task_id)
            if task["status"] == "succeeded":
                return self.results._get(db, task_id)
            _, record = self.models._resolve(db, task["model_id"], task["revision"])
            parameters = json.loads(task["parameters"])
            statistics = self._statistics(db, record, parameters)
            fault("after_compute")
            return self.results._publish(db, task, statistics, fault)
