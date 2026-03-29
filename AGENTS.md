# Agentic Coding Guidelines for LeRobot

## Project Overview
LeRobot is a PyTorch library for state-of-the-art machine learning for real-world robotics. The main source code is located in `src/lerobot/`.

## Build/Lint/Test Commands

### Installation
```bash
# Install with all dependencies
pip install -e ".[all]"

# Install development dependencies
pip install -e ".[dev]"  # Includes pre-commit, mypy, debugpy
pip install -e ".[test]"  # Includes pytest, pytest-timeout, pytest-cov
pip install -e ".[aloha]"  # For simulation testing
pip install -e ".[pusht]"  # For pusht simulation

# Install specific policy dependencies
pip install -e ".[pi]"        # Pi0, Pi0Fast, Pi05 policies
pip install -e ".[smolvla]"   # SmolVLA policy
pip install -e ".[wallx]"     # WallX policy
```

### Linting & Formatting
```bash
# Run all pre-commit hooks
pre-commit install  # One-time setup
pre-commit run --all-files

# Run ruff linter only (with auto-fix)
ruff check --fix .

# Run ruff formatter
ruff format .

# Run mypy type checker
mypy --config-file=pyproject.toml

# Run bandit security checks
bandit -c pyproject.toml -r src/lerobot

# Check for typos
typos --format=brief
```

### Testing
```bash
# Install git-lfs for test artifacts (required!)
git lfs install
git lfs pull

# Run all tests
pytest -sv ./tests

# Run a specific test file
pytest -sv tests/processor/test_act_processor.py

# Run a specific test function
pytest -sv tests/processor/test_act_processor.py::test_make_act_processor_basic

# Run tests with coverage
pytest --cov=lerobot --cov-report=term-missing

# Run tests matching a pattern
pytest -sv -k "test_act"

# Run tests with timeout (5 minutes)
pytest --timeout=300

# Run end-to-end training tests
make test-act-ete-train DEVICE=cpu
make test-diffusion-ete-train DEVICE=cpu
```

## Code Style Guidelines

### File Structure
- Source code in `src/lerobot/`
- Tests in `tests/`
- Use Apache License 2.0 header in all files (see existing files for template)
- Each Python file should start with `#!/usr/bin/env python` shebang for scripts

### Imports
```python
# Standard library imports first
from collections.abc import Callable
from typing import Any

# Third-party imports (alphabetical)
import torch
from torch.optim import Optimizer

# Local/lerobot imports (absolute imports from package root)
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.utils.logging_utils import AverageMeter

# Use 'from lerobot.xxx import yyy' not 'import lerobot.xxx'
# Exception: lerobot.__init__ uses 'from lerobot.__version__ import __version__'
```

### Formatting
- Line length: 110 characters (configured in ruff)
- Use double quotes for strings
- Use 4-space indentation (tabs not allowed)
- Trailing commas in multi-line imports and calls
- Add docstrings to all public functions, classes, and modules
- Docstring convention: Google style (see examples below)

### Naming Conventions
```python
# Classes: CapWords (PascalCase)
class AverageMeter:
class MetricsTracker:

# Functions/methods: snake_case
def update_policy():
def compute_batch_weights():

# Variables: snake_case
batch_size = 32
grad_clip_norm = 1.0

# Constants: SCREAMING_SNAKE_CASE
MAX_EPISODE_LENGTH = 1000
DEVICE = "cuda"

# Private methods/attributes: _prefixed
def _internal_method(self):
self._private_state = None

# Type variables: single capital letter or CapWords
T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., object])
```

### Type Annotations
```python
# Use type hints for function signatures
def update(self, val: float, n: int = 1) -> None:
def compute_batch_weights(self, batch: Any) -> tuple[dict, dict | None]:

# Use TypedDict for structured dictionaries
class EnvTransition(TypedDict):
    observation: RobotObservation | None
    action: PolicyAction | RobotAction | EnvAction | None

# Use Union syntax | instead of Optional/Union (Python 3.12+)
def func(x: str | None) -> int | None:

# Use collections.abc for abstract types
from collections.abc import Callable, Iterable, Sequence
def process(items: Sequence[str]) -> Iterable[int]:
```

### Dataclasses for Configuration
```python
from dataclasses import dataclass, field

@dataclass
class DatasetConfig:
    repo_id: str
    root: str | None = None
    episodes: list[int] | None = None
    image_transforms: ImageTransformsConfig = field(default_factory=ImageTransformsConfig)

    def __post_init__(self) -> None:
        # Validate fields here
        if self.episodes is not None:
            if any(ep < 0 for ep in self.episodes):
                raise ValueError("Episode indices must be non-negative")
```

### Docstrings (Google Style)
```python
def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator: Accelerator,
) -> tuple[MetricsTracker, dict]:
    """
    Performs a single training step to update the policy's weights.

    This function executes the forward and backward passes, clips gradients, and steps the optimizer and
    learning rate scheduler. Accelerator handles mixed-precision training automatically.

    Args:
        train_metrics: A MetricsTracker instance to record training statistics.
        policy: The policy model to be trained.
        batch: A batch of training data.
        optimizer: The optimizer used to update the policy's parameters.
        grad_clip_norm: The maximum norm for gradient clipping.
        accelerator: The Accelerator instance for distributed training.

    Returns:
        A tuple containing the updated MetricsTracker and a dictionary of outputs.
    """
```

### Error Handling
```python
# Use specific exception types
raise ValueError(f"Invalid episode index: {ep}")

# Validate in __post_init__ for dataclasses
def __post_init__(self) -> None:
    if self.batch_size > self.n_episodes:
        raise ValueError(
            "The eval batch size is greater than the number of eval episodes. "
            f"({self.batch_size} > {self.n_episodes}). "
            "To fix this, increase n_episodes or lower batch_size."
        )

# Use logging instead of print in production code
import logging
logging.info("Training completed")
logging.warning(f"Skipping missing file: {path}")
```

### Test Conventions
```python
# Test file naming: test_<module_name>.py
# Test function naming: test_<function_name>_<scenario>

def test_make_act_processor_basic():
    """Test basic creation of ACT processor."""
    config = create_default_config()
    preprocessor, postprocessor = make_act_pre_post_processors(config, stats)

    assert preprocessor.name == "policy_preprocessor"
    assert len(preprocessor.steps) == 4

# Use pytest fixtures from conftest.py
# Use skip decorators for platform-specific tests
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
def test_gpu_processing():
    ...

# Use temporary directories for file operations
import tempfile
with tempfile.TemporaryDirectory() as tmpdir:
    ...
```

### Module Organization
```
src/lerobot/
├── __init__.py           # Public API, available_* lists
├── __version__.py        # Version info
├── types.py              # Type aliases and TypedDicts
├── configs/              # Configuration dataclasses and CLI parser
├── policies/             # Robot learning policies (ACT, Diffusion, etc.)
├── envs/                 # Simulation environments
├── datasets/             # Dataset handling and transforms
├── robots/               # Real robot interfaces
├── cameras/              # Camera interfaces
├── motors/               # Motor controller interfaces
├── teleoperators/        # Teleoperation interfaces
├── scripts/              # CLI entry points (lerobot-train, etc.)
├── utils/                # Utility functions
├── optim/                # Optimizers and schedulers
├── processor/            # Data processing pipelines
└── rl/                   # Reinforcement learning utilities
```

### Key Dependencies
- PyTorch (>=2.2.1)
- HuggingFace libraries: diffusers, accelerate, datasets, huggingface-hub
- Gymnasium for simulation environments
- draccus for configuration parsing
- einops for tensor operations
- Ruff for linting/formatting
- mypy for type checking
