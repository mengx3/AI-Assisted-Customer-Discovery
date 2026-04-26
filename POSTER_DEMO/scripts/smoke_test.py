from __future__ import annotations
from pathlib import Path
import tempfile
import subprocess
import sys

def main():
    tmp = Path(tempfile.mkdtemp())
    raw = tmp / "customers.csv"
    outdir = tmp / "outputs"

    subprocess.check_call([sys.executable, "scripts/generate_sample_data.py", "--out", str(raw), "--n", "500"])
    subprocess.check_call([sys.executable, "scripts/run_pipeline.py", "--input", str(raw), "--outdir", str(outdir)])

    assert (outdir / "data" / "customers_cleaned.csv").exists()
    assert (outdir / "reports" / "cluster_profiles.csv").exists()
    assert (outdir / "models" / "conversion_model.joblib").exists()
    print("Smoke test passed.")

if __name__ == "__main__":
    main()
