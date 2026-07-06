import subprocess
from pathlib import Path


# Resolve adb from the project-local platform-tools bundle so training
# works regardless of whether the calling shell has activate_phase1.ps1
# on its PATH (e.g. when launched via `conda run`).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_LOCAL_ADB = _PROJECT_ROOT / "tools" / "platform-tools" / "adb.exe"
_ADB_BIN = str(_LOCAL_ADB) if _LOCAL_ADB.exists() else "adb"


class ADBController:
    def __init__(self, device_id=None):
        self.device_id = device_id

    def _adb_cmd(self):
        cmd = [_ADB_BIN]
        if self.device_id:
            cmd.extend(["-s", self.device_id])
        return cmd

    def run(self, *args):
        command = self._adb_cmd() + list(args)

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip())

        return result.stdout.strip()

    def devices(self):
        return self.run("devices")

    def shell(self, *args):
        return self.run("shell", *args)