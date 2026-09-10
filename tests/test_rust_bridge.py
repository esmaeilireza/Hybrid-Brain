"""Rust/Python bit-exact gate - delegates to benchmarks/ab_bridge.py.

One protocol, one bridge (ADR-013). This test runs the canonical A/B
harness (100 neurons x 1000 ticks, spike masks + sub-tick fracs) and
asserts its acceptance verdict via exit code.
Skips only if neither the built probe nor the cargo toolchain exists
(CI-safe); a skip on THIS machine is an open bug, not a pass.
"""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROBE = ROOT / "rust-core" / "target" / "release" / "ab_probe.exe"
BRIDGE = ROOT / "benchmarks" / "ab_bridge.py"


def _rust_available() -> bool:
    if PROBE.exists():
        return True
    return shutil.which("cargo") is not None


def test_rust_python_bit_equivalent() -> None:
    if not _rust_available():
        import pytest
        pytest.skip("Rust probe not built and toolchain absent")
    result = subprocess.run(
        [sys.executable, str(BRIDGE)],
        cwd=ROOT, capture_output=True, text=True, timeout=300,
    )
    # Print the harness's own report into pytest output on failure:
    assert result.returncode == 0, (
        f"A/B FAILED\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")
    assert "A/B ACCEPTED" in result.stdout
