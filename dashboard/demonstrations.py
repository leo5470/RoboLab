"""Demonstration API and local media delivery, independent of evaluation loaders."""
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from robolab.core.omy_postprocess.library_worker import VideoQueue


def register_demonstrations(app, catalog):
    router = APIRouter()
    queue = VideoQueue(catalog)
    stop = threading.Event()

    def refresh_loop():
        while not stop.is_set():
            try:
                catalog.refresh()
            except Exception as error:
                catalog.warnings.append(str(error))
            stop.wait(max(1, catalog.refresh_s))

    def snapshot():
        catalog.refresh()
        with catalog.lock:
            states = queue.states()
            rows = []
            for row in catalog.rows.values():
                item = catalog.public(row)
                job = states.get(row["id"], {})
                if job:
                    item["render_status"] = job["video_status"]
                    item["render_error"] = job["render_error"]
                if item["video_status"] != "ready":
                    item.update(job)
                rows.append(item)
            return rows

    @router.get("/demonstrations", include_in_schema=False)
    def page():
        return FileResponse(Path(__file__).parent / "templates/demonstrations.html")

    @router.post("/api/demonstrations/refresh")
    def refresh():
        catalog.refresh(force=True)
        return {"updated": catalog.updated, "count": len(catalog.rows)}

    @router.get("/api/demonstrations")
    def episodes(family: str = "", kind: str = "", layout: str = "", session: str = "", outcome: str = "",
                 status: str = "", review: str = "", q: str = "", offset: int = Query(0, ge=0),
                 limit: int = Query(24, ge=1, le=200)):
        rows = snapshot()
        for key, value in (("family",family),("kind",kind),("layout",layout),("session",session),
                           ("outcome",outcome),("video_status",status),("review",review)):
            if value:
                rows = [r for r in rows if r.get(key) == value]
        if q:
            rows = [r for r in rows if q.lower() in " ".join(str(v) for v in r.values()).lower()]
        rows.sort(key=lambda r: (r["video_status"] != "ready",r["family"],r["task"],r["session"],r["id"]))
        return {"episodes": rows[offset:offset+limit], "total": len(rows), "updated": catalog.updated}

    @router.get("/api/demonstrations/tasks")
    def tasks():
        rows = snapshot()
        families = {}
        for row in rows:
            item = families.setdefault(row["family"], {"family":row["family"], "collected":0, "generated":0,
                "ready":0, "native_ready":0, "held":0, "layouts":set(), "sessions":set(), "poster":None})
            item[row["kind"]] += 1
            item["ready"] += row["video_status"] == "ready"
            item["native_ready"] += (row["kind"] == "generated" and row["video_status"] == "ready"
                                     and not row.get("video_upgrade_pending", False))
            item["held"] += row["review"] == "contact_review_required"
            item["layouts"].add(row["layout"])
            item["sessions"].add(row["session"])
            item["poster"] = item["poster"] or row["poster"]
        for item in families.values():
            item["layouts"] = sorted(item["layouts"])
            item["sessions"] = sorted(item["sessions"])
        return {"tasks":sorted(families.values(),key=lambda r:r["family"]), "updated":catalog.updated,
                "warnings":catalog.warnings[-20:]}

    @router.get("/api/demonstrations/{episode_id}")
    def detail(episode_id: str):
        row = next((r for r in snapshot() if r["id"] == episode_id), None)
        if not row:
            raise HTTPException(404, "Episode is not in the active catalog")
        return row

    @router.get("/api/demonstrations/{episode_id}/{media}")
    def artifact(episode_id: str, media: str):
        if media not in ("video", "poster"):
            raise HTTPException(404)
        catalog.refresh()
        row = catalog.rows.get(episode_id)
        filename = row.get("_" + media) if row else None
        if not filename or not catalog.allowed(filename) or not Path(filename).is_file():
            raise HTTPException(404, "Media is not available")
        return FileResponse(filename, media_type="video/mp4" if media == "video" else "image/jpeg")

    app.include_router(router)
    previous_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(application):
        async with previous_lifespan(application):
            stop.clear()
            thread = threading.Thread(target=refresh_loop, daemon=True)
            thread.start()
            try:
                yield
            finally:
                stop.set()
                thread.join(timeout=5)

    app.router.lifespan_context = lifespan
