"""Oracle RPC client: spawns `node oracle/bin/oracle.js serve` and caches results by content hash (07 §10.3).

Cache key = sha256(ps_commit, oracle_git, contract_version, canonical request). The cache lives in
`tests/.oracle_cache/` (not committed; T1 rebuilds it). Every response carries `ps_commit` / `oracle_git`.

    with OracleClient() as oc:
        oc.call("fixedpoint", {"ops": [["modify", 100, 6144, 4096]]})
"""
from __future__ import annotations

import atexit
import hashlib
import json
import os
import pathlib
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import orjson

ROOT = pathlib.Path(__file__).resolve().parent.parent
ORACLE = ROOT / "oracle" / "bin" / "oracle.js"
CACHE_DIR = ROOT / "tests" / ".oracle_cache"
CONTRACT_VERSION = 1  # docs/design/contract/drawkey.md


class OracleError(RuntimeError):
    pass


class _Proc:
    """One `oracle.js serve` subprocess speaking line-delimited JSON-RPC."""

    def __init__(self) -> None:
        self.p = subprocess.Popen(["node", str(ORACLE), "serve"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, cwd=str(ROOT), bufsize=0)
        self.lock = threading.Lock()
        self.n = 0

    def call(self, method: str, params: Any) -> dict:
        with self.lock:
            self.n += 1
            self.p.stdin.write(orjson.dumps({"id": self.n, "method": method, "params": params}) + b"\n")
            self.p.stdin.flush()
            line = self.p.stdout.readline()
            if not line:
                raise OracleError(f"oracle process died: {self.p.stderr.read().decode()[-2000:]}")
            return orjson.loads(line)

    def close(self) -> None:
        if self.p.poll() is None:
            self.p.stdin.close()
            self.p.wait(timeout=10)


class OracleClient:
    """Pool of oracle subprocesses with an on-disk content-hash cache."""

    def __init__(self, workers: int = 1, cache: bool = True, cache_dir: pathlib.Path = CACHE_DIR) -> None:
        self.workers = max(1, workers)
        self.procs: list[_Proc] = []
        self.cache = cache
        self.cache_dir = cache_dir
        self._ident: tuple[str, str] | None = None
        self._rr = 0
        atexit.register(self.close)

    # -- process management
    def _proc(self) -> _Proc:
        if not self.procs:
            self.procs = [_Proc() for _ in range(self.workers)]
        self._rr = (self._rr + 1) % len(self.procs)
        return self.procs[self._rr]

    def close(self) -> None:
        for p in self.procs:
            p.close()
        self.procs = []

    def __enter__(self) -> "OracleClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- identity for cache keys
    def ident(self) -> tuple[str, str]:
        if self._ident is None:
            r = self._proc().call("ping", {})
            self._ident = (r["ps_commit"], r["oracle_git"])
        return self._ident

    def _key(self, method: str, params: Any) -> str:
        ps, og = self.ident()
        canon = orjson.dumps({"m": method, "p": params}, option=orjson.OPT_SORT_KEYS)
        return hashlib.sha256(f"{ps}|{og}|{CONTRACT_VERSION}|".encode() + canon).hexdigest()

    # -- calls
    def call(self, method: str, params: Any = None) -> Any:
        params = {} if params is None else params
        path = None
        if self.cache:
            k = self._key(method, params)
            path = self.cache_dir / k[:2] / f"{k}.json"
            if path.exists():
                return orjson.loads(path.read_bytes())
        r = self._proc().call(method, params)
        if "error" in r:
            raise OracleError(f"{method}: {r['error']}\n{r.get('stack', '')}")
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f".tmp{os.getpid()}")
            tmp.write_bytes(orjson.dumps(r["result"]))
            tmp.replace(path)
        return r["result"]

    def map(self, method: str, params_list: list[Any]) -> list[Any]:
        """Parallel calls over the process pool (order preserved)."""
        self.ident()
        if not self.procs:
            self._proc()
        with ThreadPoolExecutor(len(self.procs)) as ex:
            return list(ex.map(lambda p: self.call(method, p), params_list))


_default: OracleClient | None = None


def default_client() -> OracleClient:
    global _default
    if _default is None:
        _default = OracleClient(workers=int(os.environ.get("ORACLE_WORKERS", "4")))
    return _default


def chunks(xs: list, n: int) -> list[list]:
    return [xs[i:i + n] for i in range(0, len(xs), n)]


def dumps(x: Any) -> str:
    return json.dumps(x, sort_keys=True)
