# tm2d

`tm2d` is a Python package for GPU-accelerated 2D template matching. It contains the core algorithmic pieces: templates, CTF parameter handling, parameter grids, comparators, result containers, and execution plans.

Higher-level workflow code, file loading helpers, plotting utilities, and analysis scripts belong in `tm2d-utils` or downstream project code.

## Dependencies

Runtime dependencies are listed in `pyproject.toml`:

- Python >= 3.9
- `numpy`
- `vkdispatch==0.1.0`
- `setuptools>=59.0`

A working Vulkan/vkdispatch setup is required to run template matching.

## Installation

From a local checkout:

```bash
cd /path/to/tm2d
python -m pip install .
```

For development, install in editable mode:

```bash
cd /path/to/tm2d
python -m pip install -e .
```

To verify the installed package version:

```bash
python -m pip show tm2d
```
