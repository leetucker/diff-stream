"""Derive a focused double-storage rerun from the exact original transfer ZIP.

Preserves both earlier bundles and all returned results. This does not modify
the installed Stata commands; set type double controls their implicit gen types.
"""

import hashlib
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.stata_bjs_review import ROOT, sha256


def build():
    original = ROOT / "benchmarks/results/stata_bundles/stata-bjs-extended-validation.zip"
    destination = ROOT / "benchmarks/work/stata-bjs-precision"
    archive = ROOT / "benchmarks/results/stata_bundles/stata-bjs-precision-validation.zip"
    wanted = {"balanced_w1_a2", "complete_w1_a2", "mpdta_w0_a0"}
    with zipfile.ZipFile(original) as z:
        prefix = "diff-stream-stata-bjs-extended/"
        manifest = json.loads(z.read(prefix + "MANIFEST.json"))
        for filename, digest in manifest["sha256"].items():
            assert hashlib.sha256(z.read(prefix + filename)).hexdigest() == digest, filename
        cases = [case for case in manifest["cases"] if case["name"] in wanted]
        assert len(cases) == 3
        files = {case[key] for case in cases for key in ("python_input", "stata_input")}
        contents = {name: z.read(prefix + name) for name in files}
        script = z.read(prefix + "validate_bjs_extended.do").decode()
        # The only estimator-setting change relative to the original script.
        script = script.replace(
            "clear all\n",
            'clear all\nset type double\ndisplay "Default storage type set to double for precision validation"\n',
            1,
        )
        assert "set type double" in script
        contents["validate_bjs_extended.do"] = script.encode()
        lines = z.read(prefix + "cases.do").decode().splitlines()
        contents["cases.do"] = (
            "\n".join(line for line in lines if any(f'name("{name}")' in line for name in wanted))
            + "\n"
        ).encode()
    contents["RUN_STATA.do"] = b"do validate_bjs_extended.do\n"
    contents["README.txt"] = (
        b"Focused Stata BJS precision validation: 3 cases, 48 fits\n\n"
        b"Extract into a new folder and cd to the folder containing RUN_STATA.do.\n"
        b"Run: do RUN_STATA.do\n"
        b"Return the entire results folder to benchmarks/results/stata_bjs_precision/.\n"
        b"Keep the baseline and extended results unchanged. Use the same installed\n"
        b"did_imputation/reghdfe/ftools versions; no package updates are needed.\n\n"
        b"The only estimator-setting change is set type double after clear all.\n"
        b"Importing inputs as double does not control implicit temporary gen types\n"
        b"inside the authors' command. This rerun checks whether those temporaries\n"
        b"explain estimate differences above 1e-7 in the extended results.\n"
        b"All specifications, weights, tolerances and data come from the original\n"
        b"hashed bundle. No installed command is edited. No Python or R needed here.\n\n"
        b"Cases: balanced_w1_a2, complete_w1_a2, mpdta_w0_a0, each with the same\n"
        b"four averaging schemes, plugin/leaveout, and dynamic/simple aggregation.\n"
        b"Expected: 48 successful status rows and 154 CSVs including diagnostics.\n"
        b"Return failures and logs too; the script continues after failed fits.\n"
    )
    precision_manifest = dict(
        cases=cases,
        expected_fits=48,
        expected_csv_files=154,
        source_bundle_sha256=sha256(original),
        default_storage_type="double",
        only_estimator_setting_change="set type double",
        sha256={name: hashlib.sha256(value).hexdigest() for name, value in contents.items()},
    )
    contents["MANIFEST.json"] = (json.dumps(precision_manifest, indent=2) + "\n").encode()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for name, value in sorted(contents.items()):
            (destination / name).write_bytes(value)
            z.writestr(f"diff-stream-stata-bjs-precision/{name}", value)
    print(archive)


if __name__ == "__main__":
    build()
