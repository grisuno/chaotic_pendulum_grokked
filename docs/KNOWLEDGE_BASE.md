# Polyglot Codebase Knowledge Graph

> Generated offline by **readmenator**. Supports C, C++, Python, Go, Rust, JS/TS, Java, C#, Shell, PHP, Dart, GDScript, Nim, ASM.
> No LLMs. No tokens. Pure static analysis.

**Total Files Parsed:** 2 | **Total Symbols Extracted:** 12 | **Total Imports:** 11

## Structural Knowledge Map
```mermaid
graph TD
    classDef mod fill:#1e1e1e,stroke:#ff6666,stroke-width:2px,color:#fff;
    classDef cls fill:#2d2d2d,stroke:#4ec9b0,stroke-width:2px,color:#fff;
    classDef fn fill:#333,stroke:#dcdcaa,stroke-width:1px,color:#dcdcaa;
    classDef ext fill:#111,stroke:#666,stroke-dasharray: 5 5,color:#aaa;
    app_py["app.py (py)"]
    class app_py mod;
    app_py_generate_and_save_chaotic_pendulum_dataset["generate_and_save_chaotic_pendulum_dataset"]
    class app_py_generate_and_save_chaotic_pendulum_dataset fn;
    app_py --> app_py_generate_and_save_chaotic_pendulum_dataset
    app_py_SymplecticPredictor["SymplecticPredictor"]
    class app_py_SymplecticPredictor cls;
    app_py --> app_py_SymplecticPredictor
    app_py_train_with_hamiltonian_regularization["train_with_hamiltonian_regularization"]
    class app_py_train_with_hamiltonian_regularization fn;
    app_py --> app_py_train_with_hamiltonian_regularization
    app_py_analyze_symplectic_invariants["analyze_symplectic_invariants"]
    class app_py_analyze_symplectic_invariants fn;
    app_py --> app_py_analyze_symplectic_invariants
    app_py_null_space_surgery_chaotic["null_space_surgery_chaotic"]
    class app_py_null_space_surgery_chaotic fn;
    app_py --> app_py_null_space_surgery_chaotic
    install_sh["install.sh (sh)"]
    class install_sh mod;
    ext_numpy["numpy"]
    class ext_numpy ext;
    app_py -.->|imports| ext_numpy
    ext_torch["torch"]
    class ext_torch ext;
    app_py -.->|imports| ext_torch
    ext_torch_nn["torch.nn"]
    class ext_torch_nn ext;
    app_py -.->|imports| ext_torch_nn
    ext_torch_optim["torch.optim"]
    class ext_torch_optim ext;
    app_py -.->|imports| ext_torch_optim
    ext_sklearn_model_selection["sklearn.model_selection"]
    class ext_sklearn_model_selection ext;
    app_py -.->|imports| ext_sklearn_model_selection
    ext_matplotlib_pyplot["matplotlib.pyplot"]
    class ext_matplotlib_pyplot ext;
    app_py -.->|imports| ext_matplotlib_pyplot
    ext_tqdm["tqdm"]
    class ext_tqdm ext;
    app_py -.->|imports| ext_tqdm
    ext_os["os"]
    class ext_os ext;
    app_py -.->|imports| ext_os
    ext_scipy_stats["scipy.stats"]
    class ext_scipy_stats ext;
    app_py -.->|imports| ext_scipy_stats
    ext_scipy_integrate["scipy.integrate"]
    class ext_scipy_integrate ext;
    app_py -.->|imports| ext_scipy_integrate
    ext_warnings["warnings"]
    class ext_warnings ext;
    app_py -.->|imports| ext_warnings
```

---

## Architecture Reference

### PY (1 files)

#### `app.py`
**Path:** `app.py`

**Classs:**
- `SymplecticPredictor` (line 128) - *Architecture respecting Hamiltonian geometry with smooth activations*

**Functions:**
- `generate_and_save_chaotic_pendulum_dataset` (line 38) - *Generates double pendulum dataset and saves it to disk for reuse.
If dataset exists, loads from file unless force_regenerate=True.*
- `train_with_hamiltonian_regularization` (line 155) - *Training with physics-informed loss term and chaotic-specific thresholds*
- `analyze_symplectic_invariants` (line 254) - *Analyzes preservation of symplectic invariants in latent representations*
- `null_space_surgery_chaotic` (line 287) - *Expands weights while preserving Lyapunov exponent structure*
- `visualize_chaotic_dynamics` (line 331) - *Visualizes phase space predictions for chaotic dynamics*
- `plot_chaotic_learning_curves` (line 377) - *Plots learning curves for chaotic system training*
- `main` (line 407)
- `double_pendulum_equations` (line 63)
- `__init__` (line 130)
- `_initialize_symplectic` (line 141) - *Orthogonal initialization preserves energy manifolds*
- `forward` (line 148)

### SH (1 files)

#### `install.sh`
**Path:** `install.sh`

*No symbols extracted*
