```text
================================================================================
STRUCTURAL TRANSFER FOR CHAOTIC HAMILTONIAN SYSTEMS
Double Pendulum Dynamics with Physics-Informed Grokking
Dataset Persistence Enabled - 1800 samples optimized
================================================================================

[1] Generating or loading chaotic pendulum dataset...
Loading existing dataset from chaotic_pendulum_dataset_n1800_tmax5.0_seed42.npz
Dataset: 1800 total samples | Train: 1440 | Test: 360

[2] Training base model (128 hidden units) with Hamiltonian regularization...
TRAINING WITH HAMILTONIAN REGULARIZATION
Grokking threshold (chaotic): 0.02
Hamiltonian weight: 0.1
TRAINING:   9%|███▉                                          | 2130/25000 [00:10<02:18, 164.97it/s, mse=0.0156, test=0.0199, hamiltonian=1.2550, best=0.0199]
CHAOTIC GROKING ACHIEVED at epoch 2143!
Test MSE: 0.019947 < threshold 0.02
TRAINING:   9%|███▉                                          | 2143/25000 [00:10<01:56, 196.80it/s, mse=0.0156, test=0.0199, hamiltonian=1.2550, best=0.0199]

[3] Analyzing symplectic invariants in base model...
Angular invariance H1: 0.4036
Angular invariance H2: 0.2073
Distance preservation: 0.9944
Symplectic score: 0.9290

[4] Generating chaotic learning curves...

[5] Evaluating base model on chaotic dynamics...

[6] Executing null space surgery for chaotic manifolds...
NULL SPACE SURGERY (CHAOTIC): 128 → 256

[7] Analyzing expanded model symplectic invariants...
Angular invariance H1: 0.4036
Angular invariance H2: 0.2073
Distance preservation: 0.9944
Symplectic score: 0.9290

[8] Zero-shot evaluation of expanded model...

[9] Generating complex chaotic test set...
Generating new dataset with 1000 samples...
Integrating trajectory 1/1000...
Integrating trajectory 101/1000...
Integrating trajectory 201/1000...
Integrating trajectory 301/1000...
Integrating trajectory 401/1000...
Integrating trajectory 501/1000...
Integrating trajectory 601/1000...
Integrating trajectory 701/1000...
Integrating trajectory 801/1000...
Integrating trajectory 901/1000...
Dataset saved to chaotic_pendulum_dataset_n1000_tmax10.0_seed123.npz

[10] Evaluating on complex chaotic dynamics...

================================================================================
CHAOTIC HAMILTONIAN TRANSFER RESULTS
================================================================================
Base Model (128 units)              | MSE: 0.019947 | Grokking: YES
Expanded Model (256 units)          | MSE: 0.020133 | Zero-shot: SUCCESS
Complex Dynamics (extreme)          | MSE: 0.146538 | Transfer: LIMITED
================================================================================

CHAOTIC SYSTEM SUCCESS METRICS:
Physical algorithm grokking: ACHIEVED
Zero-shot structural transfer: SUCCESSFUL
Chaotic regime transfer: DEGRADED
Symplectic invariance: PRESERVED

CLAIM VALIDATION: STRUCTURAL TRANSFER DEMONSTRATED FOR CHAOTIC SYSTEMS

Generated artifacts:
   - chaotic_phase_space_base_model.png
   - chaotic_phase_space_expanded_model.png
   - chaotic_phase_space_complex_challenge.png
   - chaotic_learning_curves_base_model.png
   - Dataset file: chaotic_pendulum_dataset_n1800_tmax5.0_seed42.npz

```
