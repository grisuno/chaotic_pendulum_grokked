# API

## app.py

### generate_and_save_chaotic_pendulum_dataset `def generate_and_save_chaotic_pendulum_dataset(n_samples, dt, t_max, seed, force_regenerate)`
- Defined: `app.py:38`
- Doc: Generates double pendulum dataset and saves it to disk for reuse.

### train_with_hamiltonian_regularization `def train_with_hamiltonian_regularization(model, X_train, y_train, X_test, y_test, epochs, patience, grok_threshold, lambda_h)`
- Defined: `app.py:155`
- Doc: Training with physics-informed loss term and chaotic-specific thresholds

### analyze_symplectic_invariants `def analyze_symplectic_invariants(model, X_sample)`
- Defined: `app.py:254`
- Doc: Analyzes preservation of symplectic invariants in latent representations

### null_space_surgery_chaotic `def null_space_surgery_chaotic(base_model, scale_factor)`
- Defined: `app.py:287`
- Doc: Expands weights while preserving Lyapunov exponent structure

### visualize_chaotic_dynamics `def visualize_chaotic_dynamics(model, X_test, y_test, model_name)`
- Defined: `app.py:331`
- Doc: Visualizes phase space predictions for chaotic dynamics

### plot_chaotic_learning_curves `def plot_chaotic_learning_curves(history, model_name)`
- Defined: `app.py:377`
- Doc: Plots learning curves for chaotic system training

### main `def main()`
- Defined: `app.py:407`

### double_pendulum_equations `def double_pendulum_equations(t, y)`
- Defined: `app.py:63`

### __init__ `def __init__(self, input_size, hidden_size, output_size)`
- Defined: `app.py:130`

### _initialize_symplectic `def _initialize_symplectic(self)`
- Defined: `app.py:141`
- Doc: Orthogonal initialization preserves energy manifolds

### forward `def forward(self, x)`
- Defined: `app.py:148`
