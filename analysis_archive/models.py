import json
from . import validation as v
from .errors import ArchiveError, NotFound, ValidationError

TARGET_KEYS = ("definition", "representation")


def target_payload(definition):
    """Persist the canonical definition together with its explicit computation form.

    The representation keeps the ordered identity directories and numeric
    event/residence semantics; it is derived from, and checked against, the
    canonical document.  No audit row is needed to parse or calculate a revision.
    """
    states = definition["states"]
    metrics = definition["metrics"]
    representation = {
        "states": [s["id"] for s in states],
        "metrics": list(metrics),
        "residence": [[str(x) for x in s["reward"]] for s in states],
        "events": [{"id": e["id"], "source": e["source"], "target": e["target"],
                    "rate": e["rate"], "payment": [str(x) for x in e["reward"]]}
                   for e in definition["events"]],
    }
    return v.dumps({"definition": definition, "representation": representation})


class Models:
    """Active model directory.

    Format one keeps the legacy model_revisions/model_heads tables active.
    Format two keeps every revision in target_model_revisions/target_model_heads;
    the legacy tables are an unread audit copy.  Both write entry points store
    the same canonical definition, so there is one shared revision sequence.
    """

    def __init__(self, store):
        self.store = store

    def _tables(self, db):
        return ("target_model_revisions", "target_model_heads") if self.store.target(db) \
            else ("model_revisions", "model_heads")

    def _append(self, db, model_id, payload):
        revisions, heads = self._tables(db)
        row = db.execute("SELECT MAX(revision) FROM " + revisions + " WHERE model_id=?",
                         (model_id,)).fetchone()
        revision = (row[0] or 0) + 1
        db.execute("INSERT INTO " + revisions + " VALUES(?,?,?)", (model_id, revision, payload))
        db.execute("""INSERT INTO """ + heads + """ VALUES(?,?)
                   ON CONFLICT(model_id) DO UPDATE SET revision=excluded.revision""",
                   (model_id, revision))
        return revision

    def put_model(self, model_id, definition):
        model_id = v.identifier(model_id)
        definition = v.model(definition)
        with self.store.transaction() as db:
            if self.store.target(db):
                return self._append(db, model_id, target_payload(definition))
            return self._append(db, model_id, v.dumps(definition))

    def put_native_model(self, model_id, definition):
        model_id = v.identifier(model_id)
        with self.store.transaction() as db:
            if not self.store.target(db):
                raise ArchiveError("native models require archive format 2")
            payload = target_payload(v.native_model(definition))
            return self._append(db, model_id, payload)

    def _resolve(self, db, model_id, revision=None):
        v.identifier(model_id)
        if revision is not None and (type(revision) is not int or revision <= 0):
            raise ValidationError("revision must be a positive integer")
        revisions, heads = self._tables(db)
        if revision is None:
            head = db.execute("SELECT revision FROM " + heads + " WHERE model_id=?", (model_id,)).fetchone()
            if head is None:
                raise NotFound("model")
            revision = head[0]
        row = db.execute("SELECT payload FROM " + revisions + " WHERE model_id=? AND revision=?",
                         (model_id, revision)).fetchone()
        if row is None:
            raise NotFound("model revision")
        return revision, self._decode(row[0])

    @staticmethod
    def _decode(payload):
        document = json.loads(payload)
        if isinstance(document, dict) and all(key in document for key in TARGET_KEYS):
            return document
        # A format-one revision: promote it to the same in-memory record shape.
        return {"definition": document, "representation": None}

    def _canonical(self, db, model_id, revision=None):
        number, record = self._resolve(db, model_id, revision)
        return number, record["definition"]

    def get_model(self, model_id, revision=None):
        with self.store.transaction() as db:
            number, definition = self._canonical(db, model_id, revision)
        return {"model_id": model_id, "revision": number, "definition": definition}

    def revisions(self, model_id):
        v.identifier(model_id)
        revisions, _ = self._tables(self.store.connection)
        rows = self.store.connection.execute(
            "SELECT revision FROM " + revisions + " WHERE model_id=? ORDER BY revision",
            (model_id,)).fetchall()
        if not rows:
            raise NotFound("model")
        return [r[0] for r in rows]
