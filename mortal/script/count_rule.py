import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

DATA_DIR = Path('D:/Data')

def check_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if '"bakaze":"S"' in line:
                return 'south'
        return 'east'

def main():
    files = sorted(DATA_DIR.glob('*.mjson'))
    total = len(files)
    print(f'total files: {total:,}')

    east, south = 0, 0
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(check_file, f): f for f in files}
        for future in as_completed(futures):
            result = future.result()
            if result == 'east':
                east += 1
            else:
                south += 1

    print(f'east (东风局): {east:,} ({east / total * 100:.2f}%)')
    print(f'south (南风局): {south:,} ({south / total * 100:.2f}%)')

if __name__ == '__main__':
    main()
