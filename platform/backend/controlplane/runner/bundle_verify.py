"""Rerun the approved Python model on R's train and test sets and repeat every approval check against the frozen
R reference, using the same parity engine and confirmed spec as the control plane.

    python verify.py <path to the dataset>

Run it with the Python environment in environment/ (see README.md). Exits 0 when every check passes."""
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from parity import compare

HERE = Path(__file__).resolve().parent


def main(dataset: Path) -> int:
    cfg = json.loads((HERE / "verify.json").read_text())
    spec = json.loads((HERE / "spec.json").read_text())
    with open(dataset, "rb") as f:
        sha = hashlib.file_digest(f, "sha256").hexdigest()
    if sha != cfg["dataset_sha256"]:
        print(f"{dataset} is not the dataset the model was validated on (sha256 {sha}, expected "
              f"{cfg['dataset_sha256']})")
        return 1
    out = HERE / "output"
    shutil.rmtree(out, ignore_errors=True)
    with tempfile.TemporaryDirectory() as tmp:
        config = Path(tmp) / "config.json"
        config.write_text(json.dumps({"candidate": str(HERE / "model.py"), "dataset": str(dataset.resolve()),
                                      "split": str(HERE / "reference" / "split.csv"), "response": spec["response"],
                                      "out": str(out)}))
        if subprocess.run([sys.executable, "-I", str(HERE / "harness.py"), str(config)], cwd=HERE).returncode:
            print("the model failed to run; see output/error.json")
            return 1

    validation = compare(HERE / "reference", out, spec)
    for check in validation["checks"]:
        value = check["value"]
        detail = "unavailable" if value is None else f"{value:.3e} <= {check['tolerance']:.0e}"
        print(f"{'pass' if check['passed'] else 'FAIL'}  {check['name']}  {detail}")
    for note in validation["notes"]:
        print(f"note  {note}")
    return 0 if validation["passed"] else 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(main(Path(sys.argv[1])))
