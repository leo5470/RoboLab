"""Recording paths shared by generation, curation, and offline consumers."""

from pathlib import Path


def source_names(source, task):
    """Keep collection session/task names, including legacy single-run inputs."""
    source = Path(source)
    if source.parent.name == task:
        return source.parent.parent.name, task, source.name
    if source.name == task:
        return source.parent.name, task, "part_000"
    return source.name, task, "part_000"


def named_output(source, task, job_id):
    session, task, part = source_names(source, task)
    return Path(session) / task / job_id / part


def job_recording(root, job):
    """New jobs declare their location; old finalized batches remain readable."""
    root = Path(root)
    return root / job.get("output_dir", str(Path("jobs") / job["id"] / "data")) / "omy_demos.hdf5"
