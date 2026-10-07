"""Export frozen decisions to the common evaluator contract without opening gold."""
from collections import defaultdict
import json

from baseline import read_lines
from prepare import DATA, HERE, digest, freeze_json


def verify_freeze():
    out = HERE / "results"
    manifest = json.loads((out / "freeze-manifest.json").read_text())
    for name, key in [("baseline.py", "code_sha256"), ("prepare.py", "normalizer_code_sha256"),
                      ("protocol.json", "protocol_sha256")]:
        if digest((HERE / name).read_bytes()) != manifest[key]:
            raise ValueError(f"Frozen code/protocol changed: {name}")
    for name, expected in manifest["input_hashes"].items():
        if digest((DATA / name).read_bytes()) != expected:
            raise ValueError(f"Frozen input changed: {name}")
    for entry in manifest["files"]:
        if digest((HERE / entry["path"]).read_bytes()) != entry["sha256"]:
            raise ValueError(f"Frozen artifact changed: {entry['path']}")
    return manifest


def main():
    verify_freeze()
    out = HERE / "results"
    lexical = read_lines(out / "lexical-predictions.jsonl")
    methods = {
        "lexical_fixed_policy": {r["query_id"]: {"status": r["decision"], "target_ids": r["target_ids"]} for r in lexical},
        "lexical_top1_forced": {r["query_id"]: {"status": "linked", "target_ids": r["top1_target_ids"]} for r in lexical},
        "splink_fixed_policy": {},
    }
    pairs = defaultdict(list)
    for r in read_lines(out / "splink-pair-predictions.jsonl.gz"):
        pairs[r["query_id"]].append(r)
    for qid, rows in pairs.items():
        rows.sort(key=lambda r: (-r["match_probability"], r["record_id"]))
        top = rows[0]["match_probability"]
        second = rows[1]["match_probability"] if len(rows) > 1 else 0.
        status = "linked" if top >= .99 and second < .5 else "nil" if top < .1 else "review"
        methods["splink_fixed_policy"][qid] = {"status": status, "target_ids": [rows[0]["record_id"]] if status == "linked" else []}
    splits = json.loads((DATA / "partitions.json").read_text())
    files = []
    for split in ["development", "evaluation"]:
        ids = set(splits[split])
        output = {method: {qid: decisions.get(qid, {"status": "failed", "target_ids": []}) for qid in sorted(ids)}
                  for method, decisions in methods.items()}
        path = out / f"conventional-{split}.json"
        freeze_json(path, output)
        files.append({"path": str(path.relative_to(HERE)), "sha256": digest(path.read_bytes()), "queries": len(ids)})
    freeze_json(out / "export-manifest.json", {
        "contract": "method -> query_id -> {status: linked|nil|review|failed, target_ids: opaque IDs}",
        "code_sha256": digest((HERE / "export_predictions.py").read_bytes()),
        "freeze_manifest_sha256": digest((out / "freeze-manifest.json").read_bytes()),
        "files": files, "gold_opened": False, "evaluation_metrics_opened": False,
        "policies": "Exactly the predeclared lexical, forced top-1 and Splink policies; no outcome-based changes.",
    })
    print(json.dumps({"exports": files, "gold_opened": False}, indent=2))


if __name__ == "__main__":
    main()
