#!/usr/bin/env python3
"""Regenerate selected sessions from qualified human pools, then replace verified datasets."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
from pathlib import Path
import queue
import sys
import threading

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_omy_mimic_collection as batch
from complete_omy_pools import verify_pool, training_plan
from robolab.core.omy_postprocess.automatic_dataset import discover_recordings, locked, process_one, verify_final
from robolab.core.omy_postprocess.io import sha256_file
from robolab.core.omy_teleop.final_dataset import gather_candidates, choose, export
from robolab.core.omy_teleop.mimic_sources import contract
from robolab.core.omy_teleop.quality import THRESHOLDS

LIMITS = {k: v[0] for k, v in THRESHOLDS.items()}


def read(path):
    return json.loads(Path(path).read_text())


def source_id(row):
    return (str(Path(row['file']).resolve()), row['episode'])


def verify_multisource(pool, qualified):
    """Every generated row must descend from this run's qualified human pool."""
    references = {source_id(s) for s in qualified}
    used = Counter()
    for row in pool['episodes']:
        if row['kind'] != 'generated':
            continue
        origin = row.get('generation_source')
        if not origin or source_id(origin) not in references:
            raise ValueError('Generated demo has no qualified human parent')
        used[source_id(origin)] += 1
    if len(used) < 2 or sum(used.values()) != 40:
        raise ValueError('The 40 generated demos must actually use multiple human sources')
    if pool['counts_by_kind'] != {'collected': 10, 'generated': 40}:
        raise ValueError('Expected exactly 10 human and 40 generated demos')
    return [{'file': key[0], 'episode': key[1], 'generated_count': n} for key, n in sorted(used.items())]


def publish_directories(root, session, targets):
    """Resume interrupted directory moves without discarding the previous version."""
    for name, source_root, primary in targets:
        source, destination = source_root/session, primary/session
        archive = root / 'previous' / session / name
        if source.exists():
            if destination.exists():
                if archive.exists():
                    raise RuntimeError('Publication archive collision; refusing to overwrite')
                archive.parent.mkdir(parents=True, exist_ok=True)
                destination.rename(archive)
            source.rename(destination)


class Refresh:
    def __init__(self, args):
        self.args = args
        self.root = args.output.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.collected = Path('output/omy_teleop').resolve()
        self.primary_generated = Path('output/omy_mimicgen').resolve()
        self.primary_pools = Path('output/omy_final_pool50').resolve()
        self.primary_training = Path('output/omy_training').resolve()
        self.primary_cache = Path('output/omy_training_cache').resolve()
        self.progress = Path('output/omy_pool_completion/summary.json').resolve()
        self.guard = threading.Lock()
        self.state_path = self.root / 'status.json'
        self.state = read(self.state_path) if self.state_path.exists() else {'status': 'processing', 'sessions': {}}

    def status(self, session, **values):
        with self.guard:
            self.state['sessions'].setdefault(session, {}).update(values)
            batch.atomic_json(self.state_path, self.state)
            progress = read(self.progress)
            item = progress['sessions'].setdefault(session, {})
            item['replacement_status'] = self.state['sessions'][session]['status']
            item['replacement_run'] = str(self.root)
            batch.atomic_json(self.progress, progress)

    def freeze_sources(self, session):
        folder = self.root / 'generated' / session
        folder.mkdir(parents=True, exist_ok=True)
        snapshot = folder / 'source_manifest.json'
        if snapshot.exists():
            saved = read(snapshot)
        else:
            with locked(self.collected / session / '.session.lock'):
                records, family, _ = discover_recordings(self.collected, self.primary_generated, session)
                records = {k: v for k, v in records.items() if v['kind'] == 'collected'}
                rows, rejected = gather_candidates(records, family, LIMITS)
            if family == 'wooden_bowl' or len(rows) < 2:
                raise ValueError('Need at least two eligible human sources; wooden bowl is excluded')
            expected = contract(Path(rows[0]['file']).parent)
            if any(contract(Path(r['file']).parent) != expected for r in rows):
                raise ValueError('Incompatible human source contracts')
            old = verify_pool(self.primary_pools / session, 50)
            humans = {r['fingerprint'] for r in old['episodes'] if r['kind'] == 'collected'}
            if len(humans) != 10 or not humans <= {r['fingerprint'] for r in rows}:
                raise ValueError('Cannot preserve the ten selected humans')
            saved = {'family': family, 'session': session, 'human_fingerprints': sorted(humans),
                'sources': [{'file': r['file'], 'input': str(Path(r['file']).parent), 'episode': r['episode'],
                             'task': r['task'], 'family': family, 'sha256': r['source_sha256']} for r in rows],
                'rejected': rejected}
            batch.atomic_json(snapshot, saved)
        for s in saved['sources']:
            if sha256_file(s['file']) != s['sha256']:
                raise ValueError('Human source changed')
        return folder, saved

    def base_job(self, source, name, stage, demos, attempts, index):
        return {'id': name, 'family': source['family'], 'task': source['task'], 'input': source['input'],
            'episode': source['episode'], 'stage': stage, 'tier': 'multisource_refresh',
            'num_demos': demos, 'max_attempts': attempts, 'keep_failed': True,
            'seed': 2800000 + index * 1000, 'selection_seed': 3800000 + index * 1000,
            'output_dir': str(Path(source['task']) / name / Path(source['input']).name),
            'ranges': {'object_pose_range': .025, 'object_yaw_range': .0872664626, 'pose_range': .01, 'yaw_range': 0.}}

    def generate(self, session, gpu):
        self.status(session, status='qualifying_humans', gpu=gpu)
        folder, saved = self.freeze_sources(session)
        qualified = []
        for i, source in enumerate(saved['sources']):
            job = self.base_job(source, f'qualify_{i:03}', 'validate', 2, 2, i)
            job.update(sources=[source], fixed_source_index=0)
            result = batch.process_job(folder, job, gpu, LIMITS)
            if result['errors']:
                raise RuntimeError(f"Qualification setup failed: {folder / 'jobs' / job['id']}")
            if result['accepted']:
                qualified.append(source)
            self.status(session, status='qualifying_humans', qualified=len(qualified), checked=i+1,
                        human_sources=len(saved['sources']))
        if len(qualified) < 2:
            raise ValueError('Fewer than two human sources passed reproduction qualification')
        batch.atomic_json(folder / 'qualified_sources.json', {'sources': qualified})
        target = self.root / 'pools' / session
        if target.exists():
            pool = verify_pool(target, 50)
            verify_multisource(pool, qualified)
            training_plan(pool, self.collected, self.primary_generated, self.root / 'cache', 50)
            return
        # A previous invocation may already have activated the verified raw pool.
        active = self.primary_pools / session
        if read(active / 'dataset.json').get('generation_version') == self.root.name:
            verify_multisource(verify_pool(active, 50), qualified)
            return
        for i in range(100):
            records, family, reviews = discover_recordings(self.collected, self.root / 'generated', session)
            rows, rejected = gather_candidates(records, family, LIMITS)
            rows = [r for r in rows if r['kind'] == 'generated' or r['fingerprint'] in saved['human_fingerprints']]
            count = sum(r['kind'] == 'generated' for r in rows)
            self.status(session, status='generating', generated_candidates=count, qualified=len(qualified))
            if count >= self.args.candidates:
                selected, normalization = choose(rows, 50, 10)
                report = {'schema_version': 1, 'status': 'complete', 'session': session, 'family': family,
                    'selected_count': 50, 'counts_by_kind': dict(Counter(r['kind'] for r in selected)),
                    'generation_mode': 'multi_source', 'generation_version': self.root.name,
                    'qualified_human_sources': len(qualified), 'quality_review': reviews,
                    'thresholds': LIMITS, 'rejected': rejected, 'normalization': normalization,
                    'candidate_count': len(rows), 'training_ready': False}
                verify_multisource({'episodes': selected, **report}, qualified)
                export(target, selected, report)
                pool = verify_pool(target, 50)
                usage = verify_multisource(pool, qualified)
                batch.atomic_json(folder / 'selection_verification.json', {'status': 'passed', 'source_usage': usage,
                    'humans_unchanged': all(sha256_file(s['file']) == s['sha256'] for s in saved['sources'])})
                training_plan(pool, self.collected, self.primary_generated, self.root / 'cache', 50)
                self.status(session, status='pool_complete', selected_count=50, source_usage=usage)
                return
            prior = [read(p) for p in (folder / 'jobs').glob('generate_*/result.json')]
            used = sum(r['attempts'] for r in prior)
            if used >= self.args.max_attempts:
                raise RuntimeError('Additional attempt budget exhausted without enough accepted candidates')
            names = {r['id'] for r in prior}
            # Resume the first missing job deterministically, distributing target layouts.
            index = next(j for j in range(1000) if f'generate_{j:03}' not in names)
            source = qualified[index % len(qualified)]
            budget = min(24, self.args.max_attempts-used)
            job = self.base_job(source, f'generate_{index:03}', 'generate', min(8, self.args.candidates-count, budget), budget, 100+index)
            job.update(sources=qualified, selection_strategy=('nearest_neighbor_object' if index % 2 == 0
                       else 'nearest_neighbor_robot_distance'), nn_k=3)
            result = batch.process_job(folder, job, gpu, LIMITS)
            if result['errors']:
                raise RuntimeError(f"Multisource generation failed: {folder / 'jobs' / job['id']}")
        raise RuntimeError('Generation job limit exceeded')

    def activate(self, session):
        qualified = read(self.root / 'generated' / session / 'qualified_sources.json')['sources']
        staged = self.root / 'pools' / session
        pool_folder = staged if staged.exists() else self.primary_pools / session
        pool = verify_pool(pool_folder, 50)
        verify_multisource(pool, qualified)
        plan_path = self.root / 'cache' / session / 'selection.json'
        if not plan_path.exists():
            plan_path = self.primary_cache / session / 'selection.json'
        plan = read(plan_path)
        training = self.root / 'training' / session
        if not training.exists():
            training = self.primary_training / session
        final = verify_final(training, plan['options'])
        if {r['fingerprint'] for r in final['episodes']} != {r['fingerprint'] for r in pool['episodes']}:
            raise ValueError('Replacement training tensors do not match the selected pool')
        self.status(session, status='publishing')
        # Retain every previous artifact. Each move is resumable after interruption.
        publish_directories(self.root, session, (
            ('training', self.root/'training', self.primary_training),
            ('cache', self.root/'cache', self.primary_cache),
            ('pool', self.root/'pools', self.primary_pools)))
        verify_multisource(verify_pool(self.primary_pools/session, 50), qualified)
        verify_final(self.primary_training/session, plan['options'])
        with self.guard:
            progress = read(self.progress)
            progress['sessions'][session].update(status='complete', generation_mode='multi_source',
                replacement_status='complete', qualified_human_sources=len(qualified),
                training_file=str(self.primary_training/session/'training.hdf5'))
            batch.atomic_json(self.progress, progress)
            index_path = self.primary_pools / 'index.json'
            if index_path.exists():
                index = read(index_path)
                for item in index['sessions']:
                    if item['session'] == session:
                        item.update(manifest_sha256=sha256_file(self.primary_pools/session/'dataset.json'), generation_mode='multi_source')
                batch.atomic_json(index_path, index)
            verification_path = self.progress.parent / 'pool_verification.json'
            if verification_path.exists():
                verification = read(verification_path)
                verification['sessions'][session].update(
                    manifest_sha256=sha256_file(self.primary_pools/session/'dataset.json'),
                    samples=sum(r['num_samples'] for r in pool['episodes']),
                    task_layouts=len({r['task'] for r in pool['episodes']}),
                    generation_mode='multi_source', qualified_human_sources=len(qualified))
                batch.atomic_json(verification_path, verification)
        self.status(session, status='complete')

    def render(self, session, gpu, replacement=True):
        if replacement:
            self.status(session, status='processing_cameras', gpu=gpu)
            active = read(self.primary_pools/session/'dataset.json')
            if ((self.root/'previous'/session).exists() or
                    (active.get('generation_version') == self.root.name and not (self.root/'training'/session).exists())):
                self.activate(session)
                return
            process_one(session, self.collected, self.primary_generated, self.root/'training', self.root/'cache', device=f'cuda:{gpu}')
            self.activate(session)
        else:
            with self.guard:
                progress = read(self.progress); progress['sessions'][session]['status']='processing_cameras'
                batch.atomic_json(self.progress, progress)
            process_one(session, self.collected, self.primary_generated, self.primary_training, self.primary_cache, device=f'cuda:{gpu}')
            with self.guard:
                progress = read(self.progress); progress['sessions'][session].update(status='complete',
                    training_file=str(self.primary_training/session/'training.hdf5')); batch.atomic_json(self.progress, progress)

    def phase(self, tasks, function):
        pending = queue.Queue()
        for task in tasks:
            pending.put(task)
        errors = []
        def worker(gpu):
            while True:
                try:
                    task = pending.get_nowait()
                except queue.Empty:
                    return
                session = task[0] if isinstance(task, tuple) else task
                try:
                    function(task, gpu)
                except Exception as error:
                    self.status(session, status='failed', error=str(error))
                    errors.append((session, str(error)))
                    print(f'{session}: FAILED: {error}', flush=True)
        with ThreadPoolExecutor(max_workers=len(self.args.gpus)) as executor:
            list(executor.map(worker, self.args.gpus))
        return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('output/omy_multisource_refresh_20261029'))
    parser.add_argument('--session', action='append')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1])
    parser.add_argument('--candidates', type=int, default=60, help='Accepted generated candidates before selecting 40')
    parser.add_argument('--max-attempts', type=int, default=500)
    parser.add_argument('--generate-only', action='store_true')
    args = parser.parse_args()
    if args.candidates < 40 or not args.gpus or len(set(args.gpus)) != len(args.gpus) or min(args.gpus)<0:
        parser.error('Need at least 40 candidates and distinct nonnegative GPUs')
    refresh = Refresh(args)
    lock = (refresh.root / 'controller.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    sessions = args.session or [p.parent.name for p in sorted(refresh.primary_pools.glob('2026102[34]*/dataset.json'))]
    if len(sessions) != len(set(sessions)) or any(not s or '/' in s or 'wooden_bowl' in s for s in sessions):
        parser.error('Sessions must be distinct original session names excluding wooden bowl')
    options = {'sessions': sessions, 'candidates': args.candidates, 'max_attempts': args.max_attempts}
    path = refresh.root / 'options.json'
    if path.exists() and read(path) != options:
        raise ValueError('Refresh options changed; choose a fresh output root')
    batch.atomic_json(path, options)
    for session in sessions:
        refresh.status(session, status='queued', error=None)
    errors = refresh.phase(sessions, refresh.generate)
    if not args.generate_only:
        successful = [s for s in sessions if refresh.state['sessions'][s]['status'] != 'failed']
        pending = [s for s,v in read(refresh.progress)['sessions'].items() if s not in sessions and v['status']!='complete']
        errors += refresh.phase([(s, True) for s in successful] + [(s, False) for s in pending],
                                lambda task,gpu: refresh.render(task[0],gpu,task[1]))
    refresh.state['status'] = 'failed' if errors else 'pools_complete' if args.generate_only else 'complete'
    batch.atomic_json(refresh.state_path, refresh.state)
    if not errors and not args.generate_only:
        with refresh.guard:
            progress = read(refresh.progress)
            progress['status'] = 'complete' if all(v['status']=='complete' for v in progress['sessions'].values()) else 'processing'
            batch.atomic_json(refresh.progress, progress)
    return int(bool(errors))


if __name__ == '__main__':
    raise SystemExit(main())
