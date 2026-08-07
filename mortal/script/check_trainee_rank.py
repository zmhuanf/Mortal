"""统计 drain 中 trainee 的最终排名分布，按提交批次分组"""
import sys, os, gzip, json, glob
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

drain_dir = 'D:/Workspace/Mortal/mortal/mortal_v4/drain'
files = sorted(glob.glob(os.path.join(drain_dir, '*.json.gz')))

by_batch = defaultdict(list)
for f in files:
    batch = os.path.basename(f).split('_')[0]
    by_batch[batch].append(f)

batches = sorted(by_batch)
print(f'drain 文件数: {len(files)}, 批次: {batches[0]} ~ {batches[-1]}, 共 {len(batches)} 批')

def rank_stats(batch_files):
    """每局取最后一个 start_kyoku 的 scores 计算 trainee 顺位"""
    ranks = []
    for f in batch_files:
        final_scores = None
        try:
            with gzip.open(f, 'rt', encoding='utf-8') as fh:
                for line in fh:
                    ev = json.loads(line)
                    if ev.get('type') == 'start_kyoku' and ev.get('scores'):
                        final_scores = ev['scores']
        except Exception:
            continue
        if final_scores is None:
            continue
        order = sorted(range(4), key=lambda i: -final_scores[i])
        # trainee 是玩家 0，a 文件对应玩家 0
        tag = os.path.basename(f).rsplit('_', 1)[-1][0]
        pid = {'a': 0, 'b': 1, 'c': 2, 'd': 3}.get(tag)
        if pid is None:
            continue
        ranks.append(order.index(pid) + 1)
    if not ranks:
        return None
    ranks = np.array(ranks)
    pts = np.array([90, 45, 0, -135])
    return {
        'n': len(ranks),
        'rank_dist': [int((ranks == k).sum()) for k in range(1, 5)],
        'avg_rank': float(ranks.mean()),
        'avg_pt': float(pts[ranks - 1].mean()),
    }

sample = batches[::max(1, len(batches) // 6)][:6]
for b in sample:
    st = rank_stats(by_batch[b])
    if st:
        print(f'batch {b}: n={st["n"]} rank={st["rank_dist"]} avg_rank={st["avg_rank"]:.3f} avg_pt={st["avg_pt"]:.1f}')
