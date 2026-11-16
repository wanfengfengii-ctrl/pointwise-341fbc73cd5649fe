"""Explicit format-one to format-two archive upgrade.

The upgrade is one writer transaction.  It copies every model revision into the
target directory (with an explicit computation representation), then rebuilds
task and result storage so that task bindings reference the *target* revision
rows.  The historical tables stay behind as audit data.  Nothing is committed
until all models, bindings and the format marker are in place, so a failure at
the documented callbacks leaves the old database fully usable.
"""
import json

from . import validation as v
from .errors import ArchiveError
from .models import target_payload
from .storage import RESULT_SCHEMA, TARGET_SCHEMA, TASK_SCHEMA


def _copy_models(db):
    rows = db.execute(
        "SELECT model_id, revision, payload FROM model_revisions ORDER BY model_id, revision").fetchall()
    for row in rows:
        definition = v.model(json.loads(row["payload"]))
        db.execute("INSERT INTO target_model_revisions VALUES(?,?,?)",
                   (row["model_id"], row["revision"], target_payload(definition)))
    for head in db.execute("SELECT model_id, revision FROM model_heads").fetchall():
        db.execute("INSERT INTO target_model_heads VALUES(?,?)", (head["model_id"], head["revision"]))


def _rebind(db):
    # Renaming results first keeps the audit tables' mutual foreign keys intact
    # while ALTER TABLE rewrites their REFERENCES clauses.
    db.execute("ALTER TABLE results RENAME TO legacy_results_audit")
    db.execute("ALTER TABLE tasks RENAME TO legacy_tasks_audit")
    for statement in TASK_SCHEMA + RESULT_SCHEMA:
        db.execute(statement)
    db.execute("""INSERT INTO tasks(task_id,request_id,binding,model_id,revision,parameters,status)
               SELECT task_id,request_id,binding,model_id,revision,parameters,status
               FROM legacy_tasks_audit""")
    db.execute("""INSERT INTO results(task_id,payload,receipt)
               SELECT task_id,payload,receipt FROM legacy_results_audit""")


def upgrade(archive, *, fault=None):
    """Switch ``archive`` from format 1 to format 2 in a single transaction."""
    if fault is None:
        def fault(stage):
            return None
    if archive.store.format() == "2":
        return
    with archive.store.transaction() as db:
        if archive.store.format(db) != "1":
            raise ArchiveError("unsupported archive format")
        for statement in TARGET_SCHEMA:
            db.execute(statement)
        _copy_models(db)
        fault("after_models")
        _rebind(db)
        db.execute("UPDATE archive_meta SET value='2' WHERE key='format'")
        fault("before_commit")
