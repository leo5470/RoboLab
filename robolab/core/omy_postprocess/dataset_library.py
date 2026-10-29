"""Browse frozen training selections independently of the demonstration video queue."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import threading
import time

import h5py
import numpy as np
from PIL import Image

from .library import identity, save_json
from .videos import _probe

CAMERAS = ('over_shoulder_left_camera', 'wrist_cam')


def inside(path, root):
    path = Path(path).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError('Dataset artifact is outside its configured root')
    return path


def read(path):
    return json.loads(Path(path).read_text())


class DatasetCatalog:
    def __init__(self, pools, training, cache, demos, refresh_s=30):
        self.pools, self.training, self.cache = (Path(p).resolve() for p in (pools, training, cache))
        self.demos, self.refresh_s = demos, refresh_s
        self.state = demos.state_dir / 'dataset_previews'
        self.lock = threading.RLock()
        self.updated = 0
        self.sessions, self.rows, self.warnings = {}, {}, []
        self.encoder = ThreadPoolExecutor(max_workers=1, thread_name_prefix='dataset-preview')
        self.jobs = {}

    def refresh(self, force=False):
        with self.lock:
            if not force and time.time() - self.updated < self.refresh_s:
                return
            self.demos.refresh()
            with self.demos.lock:
                media = {(r.get('_canonical', r['_source']), r['_group']): dict(r)
                         for r in self.demos.rows.values()}
            images = {}
            paths = list(self.cache.glob('*/rendered/*/*/manifest.json'))
            paths += list((self.demos.state_dir / 'renders').glob('*/*/manifest.json'))
            paths += [Path(r['_images']).parent / 'manifest.json' for r in media.values() if r.get('_images')]
            for path in set(paths):
                if any(p.startswith('.') for p in path.parts):
                    continue
                try:
                    if not path.resolve().is_relative_to(self.cache) and not self.demos.allowed(path):
                        continue
                    m = read(path); p = m.get('profile', {})
                    if (m.get('status') == 'rendered_for_review' and p.get('camera_scale') == 1
                            and p.get('camera_width') is None and p.get('camera_height') is None
                            and p.get('renderer') == 'realtime' and p.get('renders_per_frame') == 2):
                        image = inside(path.parent / 'images.hdf5', path.parent)
                        if image.is_file():
                            images[(p.get('source_fingerprint'), p.get('config_sha256'))] = (str(image), p['source_group'])
                except (OSError, ValueError, KeyError):
                    continue
            sessions, rows, warnings = {}, {}, []
            try:
                progress = read(self.pools.parent / 'omy_pool_completion/summary.json').get('sessions', {})
            except (OSError, ValueError):
                progress = {}
            for path in sorted(self.pools.glob('*/dataset.json')):
                try:
                    inside(path, self.pools)
                    pool = read(path); session = path.parent.name; episodes = pool['episodes']
                    counts = Counter(r['kind'] for r in episodes)
                    if (pool.get('status') != 'complete' or pool.get('session') != session
                            or pool['selected_count'] != 50 or len(episodes) != 50
                            or len({r['fingerprint'] for r in episodes}) != 50
                            or counts != pool['counts_by_kind'] or counts['collected'] > 10
                            or set(counts) - {'collected', 'generated'}):
                        raise ValueError('Expected a complete, unique 50-demo selection with at most 10 humans')
                    exported, training_error = {}, None
                    final = self.training / session / 'dataset.json'
                    training_file = inside(final.parent / 'training.hdf5', self.training)
                    if final.is_file():
                        try:
                            manifest = read(inside(final, self.training))
                            exported = {r['fingerprint']: r for r in manifest['episodes']}
                            if (manifest.get('status') != 'complete' or manifest.get('selected_count') != 50
                                    or len(manifest['episodes']) != 50 or not training_file.is_file()
                                    or set(exported) != {r['fingerprint'] for r in episodes}
                                    or len({r['group'] for r in exported.values()}) != 50
                                    or any(not re.fullmatch(r'data/demo_\d+', r['group']) for r in exported.values())
                                    or any(any(exported[r['fingerprint']][k] != r[k] for k in ('kind', 'task', 'num_samples')) for r in episodes)):
                                raise ValueError('Training export does not match the frozen selection')
                        except (OSError, ValueError, KeyError) as error:
                            exported, training_error = {}, str(error)
                    group_rows = {}
                    for number, ep in enumerate(episodes):
                        source = inside(path.parent / ep['file'], path.parent)
                        if not source.is_file() or ep['family'] != pool['family'] or ep['session'] != session:
                            raise ValueError('Missing or mismatched selected recording')
                        key = identity(path, ep['fingerprint'])
                        base = self.demos.base(source, 'data/' + ep['episode'], ep['kind'], pool['family'], ep['task'], session)
                        base.update(id=key, number=number + 1, fingerprint=ep['fingerprint'],
                                    sample_count=ep['num_samples'], duration_s=ep['num_samples']/15,
                                    training_group=None, _video=None, _poster=None, _images=None,
                                    _manifest=str(path), generation_source=ep.get('generation_source'))
                        old = media.get((ep['source_file'], 'data/' + ep['episode']))
                        if old:
                            for field in ('_video', '_poster'):
                                base[field] = old.get(field)
                        if exported:
                            base.update(_images=str(training_file), _image_group=exported[ep['fingerprint']]['group'],
                                        _image_prefix='obs/images', training_group=exported[ep['fingerprint']]['group'])
                        elif (ep['fingerprint'], ep['config_sha256']) in images:
                            image, image_group = images[(ep['fingerprint'], ep['config_sha256'])]
                            if image_group.strip('/') == 'data/' + ep['episode']:
                                base.update(_images=image, _image_group=image_group, _image_prefix='obs/image_obs')
                        if base['_images']:
                            st = Path(base['_images']).stat()
                            revision = hashlib.sha256(f"v1:{base['_images']}:{st.st_size}:{st.st_mtime_ns}:{base['_image_group']}".encode()).hexdigest()[:20]
                            base['_preview'] = str(self.state / key / revision)
                        group_rows[key] = base
                    ready = sum(bool(r['_images']) for r in group_rows.values())
                    current = progress.get(session, {})
                    training_error = training_error or current.get('error')
                    sessions[session] = dict(session=session, family=pool['family'], selected_count=50,
                        replacement_status=current.get('replacement_status'),
                        generation_mode=pool.get('generation_mode'),
                        qualified_human_sources=pool.get('qualified_human_sources'),
                        layouts=sorted({r['layout'] for r in group_rows.values()}),
                        collected=counts['collected'], generated=counts['generated'], cameras_ready=ready,
                        export_status='complete' if exported else 'error' if training_error else
                                      'processing' if current.get('status') == 'processing_cameras' else 'pending',
                        export_error=training_error, training_ready=bool(exported and manifest.get('training_ready')),
                        manifest=f'/api/datasets/{session}/manifest',
                        training_manifest=f'/api/datasets/{session}/training-manifest' if exported else None,
                        download=f'/api/datasets/{session}/download' if exported else None)
                    rows.update(group_rows)
                except (OSError, ValueError, KeyError, TypeError) as error:
                    warnings.append(f'{path.parent.name}: {error}')
            self.sessions, self.rows, self.warnings, self.updated = sessions, rows, warnings, time.time()

    def public(self, row):
        value = {k: v for k, v in row.items() if not k.startswith('_')}
        base = '/api/datasets/episodes/' + row['id']
        video = bool(row.get('_video')) or bool(row.get('_preview') and (Path(row['_preview']) / 'ready.json').is_file())
        value.update(cameras_ready=bool(row['_images']), video=base + '/video' if video else None,
                     poster=base + '/poster' if row.get('_poster') or row['_images'] else None,
                     frame=base + '/frame' if row['_images'] else None)
        origin = row.get('generation_source') or {}
        if origin.get('file') and origin.get('episode'):
            source_id = identity(origin['file'], 'data/' + origin['episode'].removeprefix('data/'))
            if source_id in self.demos.rows:
                value['source_url'] = '/demonstrations?demo=' + source_id
        job = self.jobs.get(row.get('_preview'))
        value['preview_status'] = ('ready' if video else 'unavailable' if not row['_images'] else
                                   'failed' if job and job.done() and job.exception() else
                                   'encoding' if job and not job.done() else 'available')
        return value

    def request_preview(self, row):
        with self.lock:
            value = self.public(row)
            if value['preview_status'] in ('ready', 'encoding', 'unavailable'):
                return value
            if sum(not f.done() for f in self.jobs.values()) >= 4:
                raise RuntimeError('Preview queue is full; try again shortly')
            self.jobs[row['_preview']] = self.encoder.submit(self.encode, dict(row))
            return self.public(row)

    @staticmethod
    def channels(file, row):
        channels = [file[f"{row['_image_group']}/{row['_image_prefix']}/{camera}"] for camera in CAMERAS]
        count, h, w, rgb = channels[0].shape
        if count != row['sample_count'] or rgb != 3 or channels[1].shape != channels[0].shape or any(c.dtype != np.uint8 for c in channels):
            raise ValueError('Camera frames do not match the selected episode')
        return channels

    def frame(self, row, step=0, width=1280):
        if not 0 <= step < row['sample_count']:
            raise ValueError('Frame is outside the selected episode')
        with h5py.File(row['_images'], 'r') as file:
            channels = self.channels(file, row)
            image = Image.fromarray(np.concatenate([c[step] for c in channels], axis=1))
        image.thumbnail((width, width))
        result = io.BytesIO(); image.save(result, format='JPEG', quality=85)
        return result.getvalue()

    def encode(self, row):
        destination = inside(row['_preview'], self.state)
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix='.encoding-', dir=destination.parent))
        try:
            with h5py.File(row['_images'], 'r') as file:
                channels = self.channels(file, row)
                count, h, w, _ = channels[0].shape
                command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-n', '-f', 'rawvideo',
                    '-pix_fmt', 'rgb24', '-s:v', f'{w*2}x{h}', '-framerate', '15', '-i', 'pipe:0', '-an',
                    '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-threads', '2', '-pix_fmt', 'yuv420p',
                    '-movflags', '+faststart', str(staging / 'video.mp4')]
                with (staging / 'encode.log').open('wb') as log:
                    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)
                    try:
                        for i in range(count):
                            process.stdin.write(np.concatenate([c[i] for c in channels], axis=1).tobytes())
                        process.stdin.close()
                        if process.wait(timeout=60):
                            raise RuntimeError('Preview encoding failed')
                    finally:
                        if process.poll() is None:
                            process.kill(); process.wait()
            stream = _probe(staging / 'video.mp4', row['sample_count'])
            save_json(staging / 'ready.json', {'fingerprint': row['fingerprint'], 'stream': stream})
            staging.rename(destination)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
