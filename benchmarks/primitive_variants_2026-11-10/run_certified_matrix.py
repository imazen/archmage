#!/usr/bin/env python3
"""Run one frozen local matrix; retain failed admission packets and live stage.

Build/freeze inputs first. This driver never retries a cell or relaxes its gate.
Run from the independent source checkout. Heavy work is scoped per measurement;
zenbench, rather than this driver, owns benchmark admission and serialization.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(sys.argv[1]).resolve()
study = sys.argv[2]
assert study in ("first", "cold")
assert (root / "SOURCE.json").is_file()
assert not (root / "local-progress.json").exists()
claim = Path(".workongoing")
assert "codex-portability-fma" in claim.read_text()
wrapper = Path.home() / "work/claudehints/scripts/run-heavy"
launcher = Path(__file__).resolve().parent / "run_local.py"
cells = [("zen5/" + tier, root / "zen5/archmage-primitive-study",
          ["--recip-qualification", "--qual-tier", tier], {})
         for tier in ("scalar", "v3", "v4")]
cells += [("wasm/" + mode, root / "host/recip_wasm_host", [],
           dict(ARCHMAGE_WASM_MODE=mode,
                ARCHMAGE_WASM_MODULE=str(root / "wasm-guest" /
                                        ("strict.wasm" if mode == "strict" else "relaxed.wasm"))))
          for mode in ("strict", "fusing", "nonfusing")]
completed = []
for name, binary, args, extra in cells:
    output = root / (name + ".json.gz")
    log = root / (name + ".log")
    output.parent.mkdir(exist_ok=True)
    assert not output.exists() and not log.exists(), name
    env = dict(os.environ, TMPDIR=str(Path.home() / "tmp"),
               RUN_HEAVY_SYSTEM_SCOPE="1", ZENBENCH_GATE_TRACE="1",
               ARCHMAGE_PRIMITIVE_RESULTS=str(output), **extra)
    command = [str(wrapper), "--mem", "8G", "--jobs", "2", "--",
               sys.executable, "-B", str(launcher), str(binary), *args,
               "--cold-study" if study == "cold" else "--certified-study"]
    with log.open("w") as stream:
        process = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
        while True:
            now = datetime.now(timezone.utc).isoformat()
            claim.write_text(now + " codex-portability-fma frozen reciprocal " + name + "\n")
            (root / "local-progress.json").write_text(json.dumps(dict(
                stage=name, wrapper_pid=process.pid, command=command,
                last_check_utc=now, completed=completed), indent=2) + "\n")
            try:
                code = process.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                pass
    completed.append(dict(context=name, returncode=code, packet_present=output.is_file()))
    print(completed[-1], flush=True)
(root / "local-progress.json").write_text(json.dumps(dict(
    stage="complete", completed=completed,
    last_check_utc=datetime.now(timezone.utc).isoformat()), indent=2) + "\n")
