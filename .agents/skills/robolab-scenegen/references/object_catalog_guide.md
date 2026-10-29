# Resolving assets

Read the current `assets/objects/object_catalog.json`; its entries and counts can
change. Entries provide `name`, `usd_path`, `dims`, and descriptive/class fields.
Paths are normally relative to the repository root. A name is not a unique asset
identifier: distinguish duplicate names by catalog index and payload path.

Use descriptions/classes to choose candidates, then inspect the actual USD.
Catalog dimensions help selection, but do not encode the root pivot, intended
root scale overrides, stable poses, collision proxy, cavity, or grasp regions.
Do not invent those fields or silently substitute similarly named assets.

Fixtures may live under `assets/fixtures/` and are not necessarily in the object
catalog. For example, resolve `grey_bin`'s actual payload and scale before planning
space around it. Never copy assumed dimensions from an old manifest as ground truth.

For custom assets, record the supplied path, selected payload/default prim, units,
scale, and measured geometry. Use unique instance names in the scene and retain
the asset identity separately. Set relative paths from the **new scene layer's**
directory, including inherited base assets.

Refresh the catalog only when requested or required for the current task, using
`assets/objects/_utils/generate_catalog.py` in an environment with its dependencies.
Do not overwrite or globally rescale assets to solve a scene-specific layout issue.
