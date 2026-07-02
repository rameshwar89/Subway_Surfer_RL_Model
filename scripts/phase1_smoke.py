from __future__ import annotations

import importlib
import shutil
import subprocess
import sys


PYTHON_PACKAGES = [
    "cv2",
    "numpy",
    "mss",
    "adbutils",
    "pyautogui",
    "pydirectinput",
    "matplotlib",
    "gymnasium",
]

COMMANDS = ["adb", "scrcpy"]


def check_imports() -> bool:
    ok = True
    for package in PYTHON_PACKAGES:
        try:
            module = importlib.import_module(package)
            version = getattr(module, "__version__", "")
            print(f"[OK] import {package} {version}".rstrip())
        except Exception as exc:
            ok = False
            print(f"[FAIL] import {package}: {exc}")
    return ok


def check_commands() -> bool:
    ok = True
    for command in COMMANDS:
        path = shutil.which(command)
        if not path:
            ok = False
            print(f"[FAIL] command {command}: not found on PATH")
            continue

        result = subprocess.run([command, "--version"], capture_output=True, text=True)
        status = "OK" if result.returncode == 0 else "FAIL"
        first_line = (result.stdout or result.stderr).strip().splitlines()[0]
        print(f"[{status}] command {command}: {first_line}")
        ok = ok and result.returncode == 0
    return ok


def main() -> int:
    imports_ok = check_imports()
    commands_ok = check_commands()
    if imports_ok and commands_ok:
        print("Phase 1 smoke check passed.")
        return 0
    print("Phase 1 smoke check incomplete. Fix the failures above, then rerun.")
    return 1


if __name__ == "__main__":
    sys.exit(main())

