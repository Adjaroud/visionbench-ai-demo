"""Construit ou interroge le catalogue SQLite local des benchmarks."""

import argparse
import hashlib
import json
import math
import sqlite3
from contextlib import closing
from pathlib import Path
from statistics import median


SOURCES = ("j2_latency_pilot.json", "j2_coco_eval/metrics.json")
SCHEMA = """
CREATE TABLE campaign (
    id TEXT PRIMARY KEY,
    scope TEXT NOT NULL,
    source_path TEXT NOT NULL UNIQUE,
    source_sha256 TEXT NOT NULL,
    image_count INTEGER NOT NULL CHECK (image_count > 0),
    protocol_json TEXT NOT NULL,
    environment_json TEXT NOT NULL
);
CREATE TABLE measurement (
    campaign_id TEXT NOT NULL REFERENCES campaign(id),
    model TEXT NOT NULL,
    weights_sha256 TEXT NOT NULL,
    metric TEXT NOT NULL,
    value REAL NOT NULL,
    unit TEXT NOT NULL,
    PRIMARY KEY (campaign_id, model, metric)
);
"""


def measurements(report, model):
    if report["kind"] == "coco2017_bbox_subset":
        if report["protocol"]["split"] != "eval":
            raise ValueError("Le catalogue de référence exige le split eval.")
        for metric, value in report["models"][model]["metrics"].items():
            if not 0 <= value <= 1:
                raise ValueError(f"Score COCO invalide pour {model}/{metric}")
            yield metric, value, "fraction"
        return

    if report["kind"] != "j2_pilot_native_pipelines":
        raise ValueError("Type de rapport inconnu.")
    sessions = [session for session in report["sessions"] if session["model"] == model]
    times = sorted(row["pipeline_ms"] for session in sessions for row in session["samples"])
    yield "pipeline_median", median(times), "ms"
    yield "pipeline_p95", times[math.ceil(0.95 * len(times)) - 1], "ms"
    yield "peak_pytorch_allocated", max(s["peak_pytorch_allocated_mib"] for s in sessions), "MiB"


def build_database(reports, database):
    campaigns = []
    rows = []
    for relative_path in SOURCES:
        raw = (reports / relative_path).read_bytes()
        report = json.loads(raw)
        digest = hashlib.sha256(raw).hexdigest()
        campaigns.append((digest, report["kind"], relative_path, digest, len(report["images"]),
                          json.dumps(report["protocol"]), json.dumps(report["environment"])))
        for model, details in report["models"].items():
            rows.extend((digest, model, details["weights_sha256"], metric, value, unit)
                        for metric, value, unit in measurements(report, model))

    # Réserver un nouveau chemin pour éviter d'écraser une base existante.
    with database.open("xb"):
        pass
    try:
        with closing(sqlite3.connect(database)) as connection, connection:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.executescript(SCHEMA)
            connection.executemany("INSERT INTO campaign VALUES (?, ?, ?, ?, ?, ?, ?)", campaigns)
            connection.executemany("INSERT INTO measurement VALUES (?, ?, ?, ?, ?, ?)", rows)
    except (sqlite3.Error, OSError):
        database.unlink(missing_ok=True)
        raise


def query(database, metric):
    with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute("""
            SELECT m.model, m.metric, m.value, m.unit, c.scope, c.image_count,
                   c.source_path, c.source_sha256, m.weights_sha256
            FROM measurement AS m JOIN campaign AS c ON c.id = m.campaign_id
            WHERE m.metric = ?
            ORDER BY m.model, c.id
        """, (metric,))
        return [dict(row) for row in rows]


def check_comparability(left, right):
    """Autorise une comparaison au sein d'une même campagne et d'une même métrique."""
    reasons = []
    if left["metric"] != right["metric"] or left["unit"] != right["unit"]:
        reasons.append("Les mesures n'ont pas la même métrique ou unité.")
    if left["source_sha256"] != right["source_sha256"]:
        reasons.append("Les mesures proviennent de campagnes différentes.")
    return {"comparable": not reasons, "reasons": reasons, "scope": left["scope"] if not reasons else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--reports", type=Path, default=Path("reports"))
    build.add_argument("--db", type=Path, required=True)
    read = commands.add_parser("query")
    read.add_argument("--db", type=Path, required=True)
    read.add_argument("--metric", required=True)
    compare = commands.add_parser("compare")
    compare.add_argument("--db", type=Path, required=True)
    compare.add_argument("--metric", required=True)
    args = parser.parse_args()
    if args.command == "build":
        build_database(args.reports, args.db)
        return
    rows = query(args.db, args.metric)
    if args.command == "compare":
        if len(rows) != 2 or rows[0]["model"] == rows[1]["model"]:
            parser.error("La comparaison attend une mesure pour chacun des deux modèles.")
        rows = {"verdict": check_comparability(*rows), "measurements": rows}
    print(json.dumps(rows, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
