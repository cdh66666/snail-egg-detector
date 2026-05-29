#!/usr/bin/env python3
import importlib.util
import pathlib


def main():
    spec = importlib.util.find_spec("tpu_mlir")
    if spec is None or spec.origin is None:
        raise SystemExit("tpu_mlir is not installed")

    root = pathlib.Path(spec.origin).resolve().parent
    target = root / "python" / "utils" / "mlir_shell.py"
    text = target.read_text()

    marker = "_codex_subprocess_env_without_bundled_libc"
    if marker not in text:
        text = text.replace(
            "import ctypes\n",
            "import ctypes\n\n\n"
            "def _codex_subprocess_env_without_bundled_libc():\n"
            "    env = os.environ.copy()\n"
            "    env.pop(\"LD_LIBRARY_PATH\", None)\n"
            "    return env\n",
        )

    text = text.replace(
        "process = subprocess.Popen(cmd_str,\n"
        "                                       shell=True,\n"
        "                                       stdout=subprocess.DEVNULL,\n"
        "                                       stderr=subprocess.DEVNULL)",
        "process = subprocess.Popen(cmd_str,\n"
        "                                       shell=True,\n"
        "                                       stdout=subprocess.DEVNULL,\n"
        "                                       stderr=subprocess.DEVNULL,\n"
        "                                       env=_codex_subprocess_env_without_bundled_libc())",
    )
    text = text.replace(
        "process = subprocess.Popen(cmd_str, shell=True)",
        "process = subprocess.Popen(cmd_str, shell=True, env=_codex_subprocess_env_without_bundled_libc())",
    )

    target.write_text(text)
    print(f"patched {target}")


if __name__ == "__main__":
    main()
