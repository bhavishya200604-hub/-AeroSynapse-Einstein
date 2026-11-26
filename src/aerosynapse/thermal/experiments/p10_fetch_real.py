"""Phase 10.1: download the real soil-temperature data (NOAA USCRN hourly product) named in configs/real/soil.yaml.

    python -m aerosynapse.thermal.experiments.p10_fetch_real [--config my.yaml] [--set data.year=2022] [--force]

Writes the station files, headers.txt and readme.txt to data.folder (inside the git-ignored data/ folder) and
records URL, size, SHA-256 and retrieval time of every file in results/p10_fetch_<tag>/metrics.json, so that the
analysis can check it reads exactly the bytes that were downloaded. Files already present with a recorded hash are
not downloaded again (--force downloads anyway). The data are public domain (U.S. Government, NOAA NCEI).
"""

from __future__ import annotations

import argparse
import hashlib
import time
import urllib.request
from pathlib import Path

import yaml

from aerosynapse.thermal.data.spec import coerce_numbers, deep_merge, parse_set
from aerosynapse.thermal.utils.experiment import ROOT, Experiment, git_hash

DEFAULT = ROOT / "configs" / "real" / "soil.yaml"


def load_cfg(files: list[str], sets: list[str]) -> dict:
    cfg = yaml.safe_load(DEFAULT.read_text(encoding="utf-8"))
    for f in files:
        cfg = deep_merge(cfg, coerce_numbers(yaml.safe_load(Path(f).read_text(encoding="utf-8")) or {}))
    return deep_merge(cfg, parse_set(sets))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def urls(cfg: dict) -> list[tuple[str, Path]]:
    d = cfg["data"]
    base, folder = d["base_url"].rstrip("/"), ROOT / d["folder"]
    out = [(f"{base}/headers.txt", folder / "headers.txt"), (f"{base}/readme.txt", folder / "readme.txt")]
    for s in d["stations"]:
        name = f"CRNH0203-{d['year']}-{s}.txt"
        out.append((f"{base}/{d['year']}/{name}", folder / str(d["year"]) / name))
    return out


def fetch(url: str, dest: Path, timeout: float = 120.0) -> dict:
    dest.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    req = urllib.request.Request(url, headers={"User-Agent": "pinn-heat-transfer research (python urllib)"})
    with urllib.request.urlopen(req, timeout=timeout) as r, open(dest, "wb") as fh:
        status = r.status
        fh.write(r.read())
    return {"status": status, "seconds": round(time.time() - t0, 2)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", action="append", default=[])
    ap.add_argument("--set", dest="sets", action="append", default=[])
    ap.add_argument("--tag", default="default")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    cfg = load_cfg(a.config, a.sets)
    with Experiment(f"p10_fetch_{a.tag}", cfg) as ex:
        files = []
        for url, dest in urls(cfg):
            rec = {"url": url, "path": dest.relative_to(ROOT).as_posix()}
            if dest.exists() and not a.force:
                rec["downloaded"] = False
            else:
                rec.update(fetch(url, dest))
                rec["downloaded"] = True
            rec["bytes"] = dest.stat().st_size
            rec["sha256"] = sha256(dest)
            rec["retrieved_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(dest.stat().st_mtime))
            files.append(rec)
            print(f"{'downloaded' if rec['downloaded'] else 'present   '} {rec['bytes']:>10d} B  {rec['path']}")
        ex.metrics.update(
            {"files": files, "licence": "public domain (U.S. Government work, NOAA NCEI)", "git": git_hash()}
        )


if __name__ == "__main__":
    main()
