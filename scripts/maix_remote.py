import argparse
import os
import posixpath
import select
import shlex
import socket
import sys
import time
from pathlib import Path

import paramiko


ROOT = Path(__file__).resolve().parents[1]
MAIN_PY = ROOT / "maixcam" / "main.py"
PREVIEW_PY = ROOT / "maixcam" / "preview.py"
MODEL_DIR = ROOT / "release" / "maixcam_copy_to_device" / "root" / "models"


def model_files():
    if not MODEL_DIR.exists():
        return []
    return sorted(
        p for p in MODEL_DIR.iterdir() if p.suffix.lower() in {".mud", ".cvimodel"}
    )


def connect(args):
    host = args.host or os.environ.get("MAIXCAM_HOST")
    if not host:
        raise SystemExit("Set MAIXCAM_HOST or pass --host, for example: --host 192.168.10.107")
    user = args.user or os.environ.get("MAIXCAM_USER", "root")
    password = args.password or os.environ.get("MAIXCAM_PASSWORD", "root")

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(
        host,
        username=user,
        password=password,
        timeout=8,
        auth_timeout=8,
        banner_timeout=8,
        look_for_keys=False,
        allow_agent=False,
    )
    return ssh


def run_checked(ssh, command, timeout=20, quiet=False):
    marker = "__MAIX_REMOTE_RC__"
    wrapped = f'{command}; rc=$?; printf "\\n%s%s\\n" "{marker}" "$rc"'
    channel = ssh.get_transport().open_session(timeout=timeout)
    channel.exec_command("sh -lc " + shlex.quote(wrapped))
    out_parts = []
    err_parts = []
    deadline = time.time() + timeout
    code = None
    while time.time() < deadline:
        if channel.recv_ready():
            chunk = channel.recv(4096).decode(errors="replace")
            out_parts.append(chunk)
            if marker in "".join(out_parts):
                break
        if channel.recv_stderr_ready():
            err_parts.append(channel.recv_stderr(4096).decode(errors="replace"))
        time.sleep(0.05)
    channel.close()
    out = "".join(out_parts)
    err = "".join(err_parts)
    if marker in out:
        before, after = out.rsplit(marker, 1)
        digits = []
        for char in after:
            if char.isdigit():
                digits.append(char)
            else:
                break
        if digits:
            code = int("".join(digits))
        out = before + after[len(digits):].lstrip("\r\n")
    if code is None:
        code = 124
        err += f"\nremote command timed out after {timeout}s: {command}\n"
    if not quiet:
        if out:
            print(out, end="" if out.endswith("\n") else "\n")
        if err:
            print(err, end="" if err.endswith("\n") else "\n", file=sys.stderr)
    if code != 0:
        raise SystemExit(code)
    return out


def sftp_mkdirs(sftp, remote_dir):
    parts = [p for p in remote_dir.split("/") if p]
    path = ""
    for part in parts:
        path += "/" + part
        try:
            sftp.stat(path)
        except IOError:
            sftp.mkdir(path)


def upload_file(sftp, local, remote):
    if not local.exists():
        raise SystemExit(f"Missing local file: {local}")
    sftp_mkdirs(sftp, posixpath.dirname(remote))
    print(f"upload {local.name} -> {remote}")
    sftp.put(str(local), remote)


def probe(args):
    ssh = connect(args)
    try:
        run_checked(
            ssh,
            "echo connected; hostname; python -V; "
            "ls -lh /root/models/snail_eggs_yolov8n_* 2>/dev/null || true",
        )
    finally:
        ssh.close()


def deploy(args, ssh=None):
    close = False
    if ssh is None:
        ssh = connect(args)
        close = True
    try:
        sftp = ssh.open_sftp()
        try:
            upload_file(sftp, MAIN_PY, f"{args.remote_dir}/main.py")
            if not args.skip_models:
                for model in model_files():
                    upload_file(sftp, model, f"/root/models/{model.name}")
        finally:
            sftp.close()
    finally:
        if close:
            ssh.close()


def start_preview(args):
    ssh = connect(args)
    try:
        kill_existing(ssh)
        sftp = ssh.open_sftp()
        try:
            upload_file(sftp, PREVIEW_PY, f"{args.remote_dir}/preview.py")
        finally:
            sftp.close()
        run_checked(ssh, f"rm -f {shlex.quote(args.remote_dir)}/headless", timeout=5, quiet=True)
        run_checked(
            ssh,
            f"cd {shlex.quote(args.remote_dir)} && (nohup python preview.py > preview.log 2>&1 &) && true",
            timeout=5,
        )
        print("preview started")
    finally:
        ssh.close()


def download_tree(sftp, remote_dir, local_dir):
    local_dir.mkdir(parents=True, exist_ok=True)
    try:
        entries = sftp.listdir_attr(remote_dir)
    except IOError:
        print(f"remote directory missing: {remote_dir}")
        return
    for entry in entries:
        remote_path = posixpath.join(remote_dir, entry.filename)
        local_path = local_dir / entry.filename
        if entry.st_mode & 0o040000:
            download_tree(sftp, remote_path, local_path)
        else:
            print(f"download {remote_path} -> {local_path}")
            sftp.get(remote_path, str(local_path))


def download_debug(args):
    ssh = connect(args)
    try:
        sftp = ssh.open_sftp()
        try:
            download_tree(sftp, args.remote_debug_dir, args.local_dir)
        finally:
            sftp.close()
    finally:
        ssh.close()


def kill_existing(ssh):
    channel = ssh.get_transport().open_session(timeout=5)
    channel.exec_command(
        "sh -lc 'killall python3 2>/dev/null || true; "
        "killall python 2>/dev/null || true; sleep 1' >/dev/null 2>&1 &"
    )
    time.sleep(1.2)
    channel.close()


def stream_run(ssh, remote_dir, run_seconds):
    channel = ssh.get_transport().open_session()
    channel.get_pty()
    channel.exec_command(f"cd {remote_dir} && python main.py")
    start = time.time()
    try:
        while True:
            if channel.recv_ready():
                data = channel.recv(4096).decode(errors="replace")
                print(data, end="")
            if channel.recv_stderr_ready():
                data = channel.recv_stderr(4096).decode(errors="replace")
                print(data, end="", file=sys.stderr)
            if channel.exit_status_ready():
                return channel.recv_exit_status()
            if run_seconds > 0 and time.time() - start >= run_seconds:
                print("\n==> Stop remote app after timed run")
                channel.send("\x03")
                time.sleep(1.0)
                return 0
            select.select([channel], [], [], 0.1)
    finally:
        channel.close()


def run_app(args):
    ssh = connect(args)
    try:
        if args.kill_existing:
            kill_existing(ssh)
        run_checked(ssh, f"mkdir -p {shlex.quote(args.remote_dir)} && touch {shlex.quote(args.remote_dir)}/headless", timeout=5, quiet=True)
        code = stream_run(ssh, args.remote_dir, args.run_seconds)
        raise SystemExit(code)
    finally:
        ssh.close()


def deploy_run(args):
    ssh = connect(args)
    try:
        if args.kill_existing:
            kill_existing(ssh)
        deploy(args, ssh=ssh)
        run_checked(ssh, f"mkdir -p {shlex.quote(args.remote_dir)} && touch {shlex.quote(args.remote_dir)}/headless", timeout=5, quiet=True)
        code = stream_run(ssh, args.remote_dir, args.run_seconds)
        raise SystemExit(code)
    finally:
        ssh.close()


def install_autostart(args):
    ssh = connect(args)
    try:
        if args.kill_existing:
            kill_existing(ssh)
        app_dir = f"/maixapp/apps/{args.app_id}"
        args.remote_dir = app_dir
        deploy(args, ssh=ssh)
        command = (
            "mkdir -p /root/snail_egg "
            f"{shlex.quote(app_dir)} /root/models && "
            f"printf %s {shlex.quote(args.app_id)} > /maixapp/auto_start.txt && "
            f"rm -f /root/snail_egg/headless {shlex.quote(app_dir)}/headless && "
            f"chmod 644 {shlex.quote(app_dir)}/main.py "
            "/root/models/snail_eggs_yolov8n_*.mud "
            "/root/models/snail_eggs_yolov8n_*.cvimodel && "
            "sync && "
            "echo Installed app: $(cat /maixapp/auto_start.txt) && "
            f"ls -lh {shlex.quote(app_dir)}/main.py /root/models/snail_eggs_yolov8n_*"
        )
        run_checked(ssh, command, timeout=20)
        if args.reboot:
            channel = ssh.get_transport().open_session(timeout=5)
            channel.exec_command("/usr/sbin/reboot -f || /sbin/reboot -f || reboot -f")
            time.sleep(0.3)
            channel.close()
            print("reboot sent")
    finally:
        ssh.close()


def reboot_board(args):
    try:
        ssh = connect(args)
        channel = ssh.get_transport().open_session(timeout=5)
        channel.exec_command("/usr/sbin/reboot -f || /sbin/reboot -f || reboot -f")
        time.sleep(0.3)
        channel.close()
        ssh.close()
        print("reboot sent")
    except Exception as exc:
        print(f"reboot sent or connection dropped: {exc}")


def main():
    parser = argparse.ArgumentParser(description="Deploy and run the snail egg detector on MaixCam.")
    parser.add_argument("--host", default=None)
    parser.add_argument("--user", default=None)
    parser.add_argument("--password", default=None)
    parser.add_argument("--remote-dir", default="/root/snail_egg")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("probe")

    p_deploy = sub.add_parser("deploy")
    p_deploy.add_argument("--skip-models", action="store_true")

    p_run = sub.add_parser("run")
    p_run.add_argument("--run-seconds", type=float, default=0)
    p_run.add_argument("--no-kill", dest="kill_existing", action="store_false")
    p_run.set_defaults(kill_existing=True)

    p_deploy_run = sub.add_parser("deploy-run")
    p_deploy_run.add_argument("--skip-models", action="store_true")
    p_deploy_run.add_argument("--run-seconds", type=float, default=0)
    p_deploy_run.add_argument("--no-kill", dest="kill_existing", action="store_false")
    p_deploy_run.set_defaults(kill_existing=True)

    p_autostart = sub.add_parser("install-autostart")
    p_autostart.add_argument("--app-id", default="cdh1_")
    p_autostart.add_argument("--skip-models", action="store_true")
    p_autostart.add_argument("--reboot", action="store_true")
    p_autostart.add_argument("--no-kill", dest="kill_existing", action="store_false")
    p_autostart.set_defaults(kill_existing=True)

    sub.add_parser("preview")

    p_debug = sub.add_parser("download-debug")
    p_debug.add_argument("--remote-debug-dir", default="/root/snail_egg/debug")
    p_debug.add_argument("--local-dir", type=Path, default=ROOT / "runs" / "maix_debug" / "latest")

    sub.add_parser("reboot")

    args = parser.parse_args()
    try:
        if args.command == "probe":
            probe(args)
        elif args.command == "deploy":
            deploy(args)
        elif args.command == "run":
            run_app(args)
        elif args.command == "deploy-run":
            deploy_run(args)
        elif args.command == "install-autostart":
            install_autostart(args)
        elif args.command == "preview":
            start_preview(args)
        elif args.command == "download-debug":
            download_debug(args)
        elif args.command == "reboot":
            reboot_board(args)
    except (paramiko.SSHException, socket.error, OSError) as exc:
        print(f"MaixCam connection failed: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
