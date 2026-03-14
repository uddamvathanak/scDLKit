# Install

## Standard tutorial path

For the primary documentation and notebook workflow, install scDLKit with the tutorial extras:

```bash
python -m pip install "scdlkit[tutorials]"
```

On Windows, prefer a short virtual-environment path such as `C:\venvs\scdlkit` if you install the `tutorials` extra. The bundled Jupyter dependencies can exceed Windows path-length limits when the environment path is deeply nested.

That installs:

- `scanpy`
- `jupyter`
- the core scDLKit package

The public tutorials default to a `quickstart` profile. Each notebook also includes a `full` profile for longer, more convincing qualitative runs without changing the overall workflow.

## CPU and GPU

scDLKit uses the same tutorial code on CPU and GPU. The notebooks and scripts should default to `device="auto"`, so the package will use CUDA when available and fall back to CPU otherwise.

### CPU or default install

```bash
python -m pip install "scdlkit[tutorials]"
```

### GPU install

Install the matching PyTorch build for your platform first, then install the tutorial extras:

```bash
python -m pip install "scdlkit[tutorials]"
```

Use the official PyTorch install selector for the correct CUDA or accelerator-specific command:

- <https://docs.pytorch.org/get-started/locally/>

## Minimal package install

If you only want the core library without Scanpy or notebooks:

```bash
python -m pip install scdlkit
```

Available extras:

- `scdlkit[notebook]`
- `scdlkit[scanpy]`
- `scdlkit[tutorials]`
- `scdlkit[dev]`
- `scdlkit[docs]`

## First command to run

Most users should open the Scanpy-first quickstart notebook:

```bash
jupyter notebook examples/train_vae_pbmc.ipynb
```

Then keep the same notebook and the same `device="auto"` setting for CPU or GPU. If you want a longer run, change the first config cell from `TUTORIAL_PROFILE = "quickstart"` to `TUTORIAL_PROFILE = "full"`.
