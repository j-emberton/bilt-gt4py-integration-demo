# Bilt + GT4Py integration demo

A small example of using [GT4Py Next](https://github.com/GridTools/gt4py) to run a numerical stencil within an ordinary [Bilt](https://github.com/aviation-impact-accelerator/bilt) model. Bilt handles the grid subsystem, relationships and structural analysis; GT4Py evaluates the five-point Laplacian. **No changes to Bilt's compiler are needed.**

Run the example on Python 3.15:

```sh
uv sync
uv run python diffusion_demo.py
```

`diffusion_demo.py` is the model and runnable example. `gt4py_adapter.py` handles field axis ordering and a temporary one-cell periodic halo; those details stay out of the model. It currently uses Bilt's private evaluation context, so it is deliberately a prototype rather than a stable integration interface.

The project follows Bilt's development checks:

```sh
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run pyrefly check
uv run ty check
uv run prek run --all-files
```

The tests compare the GT4Py result against an independent NumPy reference, including unequal grid spacing, reversed named axes and periodic boundaries. CI runs these checks on macOS and Linux.

## Dependencies

`pyproject.toml` tracks Bilt's `main` branch and the [Python 3.15 GT4Py fork](https://github.com/j-emberton/gt4py) on `main`. **GT4Py's fork must be updated to allow Python 3.15 before `uv sync` or CI can pass.** Once that work lands, run `uv lock` and commit `uv.lock` so the demo is reproducible. Until then, the branch dependencies are intentionally a moving target.
