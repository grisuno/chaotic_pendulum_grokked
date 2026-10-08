# API

## app.py
- `generate_and_save_chaotic_pendulum_dataset` (function) `app.py:38` `def generate_and_save_chaotic_pendulum_dataset(n_samples, dt, t_max, seed, force_regenerate)` -- Generates double pendulum dataset and saves it to disk for reuse.
- `double_pendulum_equations` (method) `app.py:63` `def double_pendulum_equations(t, y)`
- `SymplecticPredictor.__init__` (method) `app.py:130` `def __init__(self, input_size, hidden_size, output_size)`
- `SymplecticPredictor.forward` (method) `app.py:148` `def forward(self, x)`
- `SymplecticPredictor.train_with_hamiltonian_regularization` (method) `app.py:155` `def train_with_hamiltonian_regularization(model, X_train, y_train, X_test, y_test, epochs, patience, grok_threshold...` -- Training with physics-informed loss term and chaotic-specific thresholds
- `SymplecticPredictor.analyze_symplectic_invariants` (method) `app.py:254` `def analyze_symplectic_invariants(model, X_sample)` -- Analyzes preservation of symplectic invariants in latent representations
- `SymplecticPredictor.null_space_surgery_chaotic` (method) `app.py:287` `def null_space_surgery_chaotic(base_model, scale_factor)` -- Expands weights while preserving Lyapunov exponent structure
- `SymplecticPredictor.visualize_chaotic_dynamics` (method) `app.py:331` `def visualize_chaotic_dynamics(model, X_test, y_test, model_name)` -- Visualizes phase space predictions for chaotic dynamics
- `SymplecticPredictor.plot_chaotic_learning_curves` (method) `app.py:377` `def plot_chaotic_learning_curves(history, model_name)` -- Plots learning curves for chaotic system training
- `SymplecticPredictor.main` (method) `app.py:407` `def main()`
