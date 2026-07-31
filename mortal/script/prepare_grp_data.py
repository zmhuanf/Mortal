import os
import gzip
import random
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

SRC_DIR = Path('D:/Data')
DST_DIR = Path('D:/Workspace/Mortal/mortal/grp_data')
VAL_RATIO = 0.05
SEED = 42

def gzip_file(args):
    src, dst = args
    with open(src, 'r', encoding='utf-8') as f_in:
        content = f_in.read()
    with gzip.open(dst, 'wt', encoding='utf-8') as f_out:
        f_out.write(content)

def main():
    files = sorted(SRC_DIR.glob('*.mjson'))
    random.seed(SEED)
    random.shuffle(files)

    n_val = int(len(files) * VAL_RATIO)
    val_files = files[:n_val]
    train_files = files[n_val:]

    train_dir = DST_DIR / 'train'
    val_dir = DST_DIR / 'val'
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for f in train_files:
        tasks.append((f, train_dir / (f.stem + '.json.gz')))
    for f in val_files:
        tasks.append((f, val_dir / (f.stem + '.json.gz')))

    print(f'train: {len(train_files)}, val: {len(val_files)}')

    done = 0
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(gzip_file, t): t for t in tasks}
        for future in as_completed(futures):
            future.result()
            done += 1
            if done % 5000 == 0:
                print(f'{done}/{len(tasks)}')

    print(f'done: {done}/{len(tasks)}')

if __name__ == '__main__':
    main()
