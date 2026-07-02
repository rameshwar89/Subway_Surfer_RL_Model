import subprocess


class ADBController:
    def __init__(self, device_id=None):
        self.device_id = device_id

    def _adb_cmd(self):
        cmd = ["adb"]
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