"""Install the official Q.ANT Native Computing Toolkit wheel (cpu-backend) into this environment.

The toolkit is distributed as GitHub release assets, not on PyPI. This script downloads the
pinned release archive, checks its SHA-256, and installs the wheel for the current platform.
For physical hardware, install the "default" variant and the Q.ANT driver instead (see README).

Usage:  uv run python scripts/install_qant_sdk.py
"""

from __future__ import annotations

import hashlib
import platform
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

VERSION = "2.5.0"
ARCHIVE = f"qant-native-computing-toolkit-wheels-cpu-backend-v{VERSION}.zip"
URL = (
    "https://github.com/Q-ANT-GmbH/qant_native_computing_toolkit/releases/download/"
    f"v{VERSION}/{ARCHIVE}"
)
SHA256 = "0f7ca623b228b1eb6441a2d40be00f3cdc5524f2df35ab5190591fa1c946a0e7"
CACHE = Path(__file__).resolve().parent.parent / ".qant"

PLATFORM_TAGS = {
    ("Linux", "x86_64"): "linux_x86_64",
    ("Windows", "AMD64"): "win_amd64",
    ("Darwin", "arm64"): "macosx_11_0_arm64",
}


def _download() -> Path:
    CACHE.mkdir(exist_ok=True)
    target = CACHE / ARCHIVE
    if not target.exists():
        print(f"downloading {URL}")
        with urllib.request.urlopen(URL, timeout=120) as response, target.open("wb") as fh:
            shutil.copyfileobj(response, fh)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    if digest != SHA256:
        target.unlink()
        raise SystemExit(f"checksum mismatch for {ARCHIVE}: {digest}")
    return target


def _extract_wheel(archive: Path) -> Path:
    key = (platform.system(), platform.machine())
    tag = PLATFORM_TAGS.get(key)
    if tag is None:
        raise SystemExit(f"no cpu-backend wheel for {key}; supported: {sorted(PLATFORM_TAGS)}")
    name = f"qant_native_computing_toolkit-{VERSION}-cp311-abi3-{tag}.whl"
    with zipfile.ZipFile(archive) as zf:
        zf.extract(name, CACHE)
    return CACHE / name


def _install(wheel: Path) -> None:
    uv = shutil.which("uv")
    if uv:
        cmd = [uv, "pip", "install", "--python", sys.executable, str(wheel)]
    else:
        cmd = [sys.executable, "-m", "pip", "install", str(wheel)]
    subprocess.run(cmd, check=True)


def main() -> None:
    _install(_extract_wheel(_download()))
    import qant_native_computing_toolkit as qant

    print(f"installed qant_native_computing_toolkit {qant.__version__}")


if __name__ == "__main__":
    main()
