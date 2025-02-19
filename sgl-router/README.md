# SGLang Router

SGLang router is a standalone module implemented in Rust to achieve data parallelism across SGLang instances.

## User docs

Please check https://sgl-project.github.io/router/router.html

## Developer docs

### Prerequisites

- Rust and Cargo installed

```bash
# Install rustup (Rust installer and version manager)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Follow the installation prompts, then reload your shell
source $HOME/.cargo/env

# Verify installation
rustc --version
cargo --version
```

- Python with pip installed


### Build Process

#### 1. Build Rust Project

```bash
$ cargo build
```

#### 2. Build Python Binding

##### Option A: Build and Install Wheel
1. Build the wheel package:
```bash
$ pip install setuptools-rust wheel build
$ python -m build
```

2. Install the generated wheel:
```bash
$ pip install <path-to-wheel>
```

If you want one handy command to do build + install for every change you make:

```bash
$ python -m build && pip install --force-reinstall dist/*.whl
```

##### Option B: Development Mode

For development purposes, you can install the package in editable mode:

Warning: Using editable python binding can suffer from performance degradation!! Please build a fresh wheel for every update if you want to test performance.

```bash
$ pip install -e .
```

**Note:** When modifying Rust code, you must rebuild the wheel for changes to take effect.

### CI/CD Setup

The continuous integration pipeline consists of three main steps:

#### 1. Build Wheels
- Uses `cibuildwheel` to create manylinux x86_64 packages
- Compatible with major Linux distributions (Ubuntu, CentOS, etc.)
- Additional configurations can be added to support other OS/architectures
- Reference: [cibuildwheel documentation](https://cibuildwheel.pypa.io/en/stable/)

#### 2. Build Source Distribution
- Creates a source distribution containing the raw, unbuilt code
- Enables `pip` to build the package from source when prebuilt wheels are unavailable

#### 3. Publish to PyPI
- Uploads both wheels and source distribution to PyPI

The CI configuration is based on the [tiktoken workflow](https://github.com/openai/tiktoken/blob/63527649963def8c759b0f91f2eb69a40934e468/.github/workflows/build_wheels.yml#L1).
