# Subsystem: root

## app.py
- Layer: utility
- Doc: _*_ coding: utf8 _*_
- Language: py
- Symbols:
  - `generate_and_save_chaotic_pendulum_dataset` (function, line 38) `def generate_and_save_chaotic_pendulum_dataset(n_samples, dt, t_max, seed, force_regenerate)`
  - `SymplecticPredictor` (class, line 128) `class SymplecticPredictor(Module)`
  - `train_with_hamiltonian_regularization` (method, line 155) `def train_with_hamiltonian_regularization(model, X_train, y_train, X_test, y_test, epochs, patience, grok_threshold, lambda_h)`
  - `analyze_symplectic_invariants` (method, line 254) `def analyze_symplectic_invariants(model, X_sample)`
  - `null_space_surgery_chaotic` (method, line 287) `def null_space_surgery_chaotic(base_model, scale_factor)`
  - `visualize_chaotic_dynamics` (method, line 331) `def visualize_chaotic_dynamics(model, X_test, y_test, model_name)`
  - `plot_chaotic_learning_curves` (method, line 377) `def plot_chaotic_learning_curves(history, model_name)`
  - `main` (method, line 407) `def main()`
  - `double_pendulum_equations` (method, line 63) `def double_pendulum_equations(t, y)`
  - `__init__` (method, line 130) `def __init__(self, input_size, hidden_size, output_size)`
  - `_initialize_symplectic` (method, line 141) `def _initialize_symplectic(self)`
  - `forward` (method, line 148) `def forward(self, x)`

## install.sh
- Layer: utility
- Language: sh
