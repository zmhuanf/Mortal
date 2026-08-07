"""拉起并守护 client.py，进程退出后自动重启"""
from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

RESTART_DELAY = 5
LOG_FILE = Path(__file__).resolve().parent / 'daemon.log'
# client_daemon.py 不含 client.py 子串，正则不会误匹配守护进程自身
PS_DETECT = (
    "Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | "
    "Where-Object { $_.CommandLine -match 'client\\.py' } | "
    "Measure-Object | Select-Object -ExpandProperty Count"
)


def log(msg: str) -> None:
    """追加写日志并同步输出到控制台"""
    line = f'[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}'
    print(line)
    with LOG_FILE.open('a', encoding='utf-8') as f:
        f.write(line + '\n')


def find_client(start: Path) -> Path | None:
    """自脚本目录逐级向上定位 client.py，兼容开发机与部署包布局"""
    for d in (start, *start.parents):
        if (d / 'client.py').is_file():
            return d / 'client.py'
    return None


def is_client_running() -> bool:
    """检测是否已有 client.py 进程，防双开抢 GPU"""
    if sys.platform != 'win32':
        return False
    try:
        rsp = subprocess.run(['powershell', '-NoProfile', '-Command', PS_DETECT],
                             capture_output=True, text=True, timeout=10)
        return rsp.stdout.strip() not in ('', '0')
    except (OSError, subprocess.TimeoutExpired):
        return False


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    client = find_client(script_dir)
    if client is None:
        log(f'ERROR: client.py not found under {script_dir}')
        return 1

    if is_client_running():
        log('ERROR: client.py already running, exit to avoid duplicate')
        return 1

    log(f'guarding {client}, restart delay {RESTART_DELAY}s')
    proc: subprocess.Popen | None = None
    try:
        while True:
            proc = subprocess.Popen([sys.executable, 'client.py'], cwd=client.parent)
            code = proc.wait()
            log(f'client exited with code {code}, restarting in {RESTART_DELAY}s')
            time.sleep(RESTART_DELAY)
    except KeyboardInterrupt:
        # Ctrl+C 可能同时送达子进程，先确认是否仍存活再终止
        if proc and proc.poll() is None:
            proc.terminate()
        log('daemon stopped')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
