"""Compare isolated PnP replay artifacts; never changes production results."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def compare(root, reference, candidate):
    left, right = root / reference, root / candidate
    reports = [json.loads((path / 'report.json').read_text()) for path in (left, right)]
    result = {
        'reference': reference, 'candidate': candidate,
        'elapsed_s': [r['episode_elapsed_s'] for r in reports],
        'elapsed_reduction_pct': 100 * (1 - reports[1]['episode_elapsed_s'] / reports[0]['episode_elapsed_s']),
        'outcomes': [[{k: row.get(k) for k in ('env_id', 'success', 'episode_step', 'score', 'events')}
                      for row in r['results']] for r in reports],
        'memory': [r['memory'] for r in reports],
    }
    result['outcomes_equal'] = result['outcomes'][0] == result['outcomes'][1]
    traces = [json.loads((path / 'actions.json').read_text()) for path in (left, right)]
    result['actions_equal'] = traces[0] == traces[1]
    requests = [json.loads((path / 'requests.json').read_text()) for path in (left, right)]
    maps = [{(r['env_id'], r['step']): r['hashes'] for r in trace} for trace in requests]
    common = maps[0].keys() & maps[1].keys()
    result['request_counts'] = [len(trace) for trace in requests]
    result['request_order_equal'] = [list(m) for m in maps][0] == [list(m) for m in maps][1]
    result['request_hash_differences'] = {key: sum(maps[0][i][key] != maps[1][i][key] for i in common)
                                          for key in next(iter(maps[0].values()))}
    with np.load(left / 'policy_images.npz') as a, np.load(right / 'policy_images.npz') as b:
        keys = sorted(set(a.files) & set(b.files))
        diffs = [np.abs(a[key].astype(np.int16) - b[key].astype(np.int16)) for key in keys]
        result['images'] = {
            'compared': len(keys), 'exact': sum(not diff.any() for diff in diffs),
            'mean_abs_pixel_difference': float(np.mean([diff.mean() for diff in diffs])),
            'max_abs_pixel_difference': max(int(diff.max()) for diff in diffs),
        }
    datasets = {}
    with h5py.File(left / 'run_0.hdf5') as a, h5py.File(right / 'run_0.hdf5') as b:
        def check(name, dataset):
            if not isinstance(dataset, h5py.Dataset):
                return
            if name not in b:
                datasets[name] = {'missing': True}
                return
            other = b[name]
            if dataset.shape != other.shape:
                datasets[name] = {'shapes': [list(dataset.shape), list(other.shape)]}
                return
            x, y = dataset[:], other[:]
            if np.array_equal(x, y):
                return
            entry = {'different': True}
            if x.dtype.kind in 'biuf' and x.size:
                entry['max_abs_difference'] = float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64))))
            datasets[name] = entry
        a.visititems(check)
        extra = []
        b.visititems(lambda name, obj: extra.append(name) if isinstance(obj, h5py.Dataset) and name not in a else None)
    result['hdf_differing_datasets'] = datasets
    result['hdf_extra_datasets'] = extra
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    comparisons = []
    for reference, candidate in [('baseline', 'baseline_repeat'), ('baseline_repeat', 'optimized'),
                                 ('optimized', 'no_viewport'), ('optimized', 'no_allocator_cleanup')]:
        if all((args.root / mode / 'report.json').exists() for mode in (reference, candidate)):
            comparisons.append(compare(args.root, reference, candidate))
    (args.root / 'comparison.json').write_text(json.dumps(comparisons, indent=2))
    for result in comparisons:
        compact = {k: v for k, v in result.items() if k not in ('outcomes', 'hdf_differing_datasets')}
        compact['hdf_differing_dataset_count'] = len(result['hdf_differing_datasets'])
        print(json.dumps(compact))
