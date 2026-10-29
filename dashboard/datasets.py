"""Read-only dataset browsing and CPU preview preparation for selected episodes."""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, Response

from robolab.core.omy_postprocess.dataset_library import inside


def register_datasets(app, catalog):
    router = APIRouter()

    def episode(key):
        catalog.refresh()
        with catalog.lock:
            row = catalog.rows.get(key)
            if row is None:
                raise HTTPException(404, 'Episode is not in the selected dataset')
            return dict(row)

    @router.get('/dataset', include_in_schema=False)
    def page():
        return FileResponse(Path(__file__).parent / 'templates/dataset.html')

    @router.get('/api/datasets')
    def datasets():
        catalog.refresh()
        with catalog.lock:
            return {'datasets': list(catalog.sessions.values()), 'warnings': catalog.warnings, 'updated': catalog.updated}

    @router.post('/api/datasets/refresh')
    def refresh():
        catalog.refresh(force=True)
        return {'updated': catalog.updated}

    @router.get('/api/datasets/episodes')
    def episodes(session: str = '', kind: str = '', layout: str = '', q: str = '',
                 offset: int = Query(0, ge=0), limit: int = Query(24, ge=1, le=200)):
        catalog.refresh()
        with catalog.lock:
            rows = list(catalog.rows.values())
            for key, value in (('session', session), ('kind', kind), ('layout', layout)):
                if value:
                    rows = [r for r in rows if r[key] == value]
            if q:
                rows = [r for r in rows if q.lower() in f"{r['family']} {r['task']} {r['session']} {r['number']}".lower()]
            rows.sort(key=lambda r: (r['session'], r['number']))
            return {'episodes': [catalog.public(r) for r in rows[offset:offset+limit]], 'total': len(rows)}

    @router.get('/api/datasets/episodes/{key}')
    def detail(key: str):
        return catalog.public(episode(key))

    @router.post('/api/datasets/episodes/{key}/preview')
    def preview(key: str):
        try:
            return catalog.request_preview(episode(key))
        except RuntimeError as error:
            raise HTTPException(429, str(error)) from error

    @router.get('/api/datasets/episodes/{key}/{media}')
    def artifact(key: str, media: str, step: int = Query(0, ge=0)):
        row = episode(key)
        if media in ('poster', 'video'):
            filename = row.get('_' + media)
            if filename and catalog.demos.allowed(filename) and Path(filename).is_file():
                return FileResponse(filename, media_type='video/mp4' if media == 'video' else 'image/jpeg')
        if media == 'video' and row.get('_preview'):
            folder = inside(row['_preview'], catalog.state)
            if (folder / 'ready.json').is_file():
                return FileResponse(inside(folder / 'video.mp4', catalog.state), media_type='video/mp4')
        if media in ('poster', 'frame') and row['_images']:
            try:
                content = catalog.frame(row, 0 if media == 'poster' else step, 640 if media == 'poster' else 1280)
                return Response(content, media_type='image/jpeg', headers={'Cache-Control': 'private, max-age=300'})
            except (OSError, KeyError, ValueError) as error:
                raise HTTPException(404, 'Camera frame is unavailable') from error
        raise HTTPException(404, 'Preview is not available yet')

    @router.get('/api/datasets/{session}/{artifact}')
    def export(session: str, artifact: str):
        catalog.refresh()
        with catalog.lock:
            item = catalog.sessions.get(session)
        if item is None:
            raise HTTPException(404, 'Dataset is unavailable')
        if artifact == 'manifest':
            filename = inside(catalog.pools / session / 'dataset.json', catalog.pools)
            return FileResponse(filename, media_type='application/json', filename=f'{session}_selection.json')
        if item['export_status'] == 'complete' and artifact in ('download', 'training-manifest'):
            name = 'training.hdf5' if artifact == 'download' else 'dataset.json'
            filename = inside(catalog.training / session / name, catalog.training)
            return FileResponse(filename, filename=f'{session}_{name}')
        raise HTTPException(404, 'Training export is not complete yet')

    app.include_router(router)
    previous = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(application):
        async with previous(application):
            try:
                yield
            finally:
                catalog.encoder.shutdown(wait=False, cancel_futures=True)

    app.router.lifespan_context = lifespan
