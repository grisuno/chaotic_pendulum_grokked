# root

*Community 0 | 2 files | cohesion 1.00*

## Definition

This community groups 2 file(s) rooted at `root` with dominant language py (cohesion 1.00). Central symbols: `SymplecticPredictor`, `__init__`, `_initialize_symplectic`, `analyze_symplectic_invariants`, `double_pendulum_equations`, `forward`, `generate_and_save_chaotic_pendulum_dataset`, `main`. Core file: `app.py` (12 symbols). Documented purpose: Autor: Gris Iscomeback Correo electrónico: grisiscomeback[at]gmail[dot]com Fecha de creación: xx/xx/xxxx Licencia: GPL v3  Descripción:.

## Files

| File | Language | Layer | Symbols | Doc |
|------|----------|-------|---------|-----|
| `app.py` | py | utility | 12 | yes |
| `install.sh` | sh | utility | 0 | no |

## Key Symbols

- `generate_and_save_chaotic_pendulum_dataset` (function, `app.py:38`) `def generate_and_save_chaotic_pendulum_dataset(n_samples, dt, t_max, seed, force` - Generates double pendulum dataset and saves it to disk for reuse.
- `double_pendulum_equations` (method, `app.py:63`) `def double_pendulum_equations(t, y)`
- `SymplecticPredictor` (class, `app.py:128`) `class SymplecticPredictor(Module)` - Architecture respecting Hamiltonian geometry with smooth activations
- `__init__` (method, `app.py:130`) `def __init__(self, input_size, hidden_size, output_size)`
- `_initialize_symplectic` (method, `app.py:141`) `def _initialize_symplectic(self)` - Orthogonal initialization preserves energy manifolds
- `forward` (method, `app.py:148`) `def forward(self, x)`
- `train_with_hamiltonian_regularization` (method, `app.py:155`) `def train_with_hamiltonian_regularization(model, X_train, y_train, X_test, y_tes` - Training with physics-informed loss term and chaotic-specific thresholds
- `analyze_symplectic_invariants` (method, `app.py:254`) `def analyze_symplectic_invariants(model, X_sample)` - Analyzes preservation of symplectic invariants in latent representations
- `null_space_surgery_chaotic` (method, `app.py:287`) `def null_space_surgery_chaotic(base_model, scale_factor)` - Expands weights while preserving Lyapunov exponent structure
- `visualize_chaotic_dynamics` (method, `app.py:331`) `def visualize_chaotic_dynamics(model, X_test, y_test, model_name)` - Visualizes phase space predictions for chaotic dynamics
- `plot_chaotic_learning_curves` (method, `app.py:377`) `def plot_chaotic_learning_curves(history, model_name)` - Plots learning curves for chaotic system training
- `main` (method, `app.py:407`) `def main()`

## Internal vs External Edges

- Internal resolved imports (EXTRACTED): 0
- Cross-boundary resolved imports (EXTRACTED): 0

## Connections

- No cross-community bridges recorded. This community is self-contained.

## Risks

- No scoped security, taint, cycle, or layer risks.

## Open Questions

- Why do 1 file(s) lack file-level docs (e.g. `install.sh`)? What purpose do they serve?
- What would break if the most connected file in root changed?
- Should root be split, given cohesion 1.00?

## Sources

- `app.py`
- `install.sh`
