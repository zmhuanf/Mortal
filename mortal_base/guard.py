"""train.py 守护：30s 轮询，崩溃自动拉起，正常完成则退出"""

import subprocess
import sys
import time
from datetime import datetime

import psutil

TRAIN = [sys.executable, 'D:/Workspace/Mortal/mortal_base/train.py']


def train_running() -> bool:
    for p in psutil.process_iter(['cmdline']):
        try:
            cmdline = ' '.join(p.info['cmdline'] or [])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if 'train.py' in cmdline:
            return True
    return False


def main():
    child = None
    while True:
        if not train_running():
            if child is not None and child.poll() == 0:
                print(f'{datetime.now().isoformat(timespec="seconds")} train.py finished normally, guard exits')
                break
            # 崩溃（非 0 退出）时经此拉起，正常完成则不复活
            child = subprocess.Popen(TRAIN)
            print(f'{datetime.now().isoformat(timespec="seconds")} launched train.py pid={child.pid}', flush=True)
        time.sleep(30)


if __name__ == '__main__':
    main()
