PSID (Preferential Subspace Identification) and IPSID (Input Preferential Subspace Identification) are dynamic state-space modeling algorithms used to analyze neural and behavioral time-series data. These methods seek to identify models where some latent dynamics (states) specifically encode certain target outputs, such as behavior (`z`), while optionally excluding unrelated neural signals (`y`) or external inputs (`u`).

### Key Training Process of PSID/IPSID

1. **Data requirements:**
   - PSID requires:
     - `y`: Neural activity (e.g., signals from brain recordings).
     - `z`: Output/behavior data (e.g., hand position, velocity, behavioral outcomes).
   - IPSID extends PSID with the inclusion of:
     - `u`: External input (e.g., sensory stimuli, environmental/task cues).
     
   Data must be formatted as `time x dimension` matrices (first dim = time samples, second = signal dim). Discontinuous trial data can be passed as lists of matrices.

2. **Zero-mean preprocessing:**
   Input signals (`y`, `z`, and `u`) are required to have zero mean for the modeling to work correctly. If not:
   - The mean is removed internally during training.
   - Mean removal is reversible when making predictions.

3. **State-space model structure:**
   PSID/IPSID learns state-space models of the form:
```
x(k+1) = A * x(k) + B * u(k) + w(k)  # Dynamics
   y(k)   = C_y * x(k) + D_y * u(k) + v(k)  # Outputs (neural activity)
   z(k)   = C_z * x(k) + D_z * u(k) + ε(k)  # Outputs (behavior)
   x(k)   = [x1(k); x2(k); x3(k)], where parts of x encode y-z relevance, y-only, z-only states.
```

   Here:
   - `x(k)`: Latent states.
   - `A`, `B`, `C`, `D`: Matrices to be learned.
   - `w(k), v(k), ε(k)`: Model noise terms.

4. **Training workflow:**
   - Stage 1: Derive `x1` (latent states that relate neural activity `y` to behavior `z`).
     - Uses an oblique projection of `z` combined with `y` onto the latent state space.
   - Stage 2: Complement `x1` with other intrinsic neural dimensions (`x2`), which capture dynamics in neural signals unrelated to outputs.
   - Optional Step 3 (IPSID): With external inputs `u`, further dissociate input-driven and intrinsic latent components. This step identifies additional latent states (`x3`) corresponding only to external input effects.

5. **Output of training:**
   A learned model encapsulated in a Python object (e.g., an `LSSM` instance), which contains parameters like:
   - `A`, `B`: State transition matrices.
   - `C_y`, `C_z`: Observation matrices.
   - `Q`, `R`: Covariance matrices of process and observation noise.

---

### Validation and Testing

1. **Cross-validation:**
   While PSID/IPSID inherently prioritize dimensions minimizing overfitting via low-dimensional latent state modeling, cross-validation (CV) is still essential to assess generalizability. Key points:
   - Perform grid search over hyperparameters (`nx` = state-space dimension, `n1` = behaviorally relevant states, `i` = horizon).
   - Split training data into `k`-folds or hold out a portion for validation during grid search.

   Validation aims to maximize metrics like:
   - **Behavior decoding R²**: Measures how well `z` (behavior) predictions align with true values.
   - **Neural self-prediction R²**: Measures prediction accuracy of neural signals `y`.

2. **Testing and performance measures:**
   On a test set, trained models are evaluated using:
   - **R² (explained variance)**: Captures behavioral decoding accuracy.
   - **Correlation coefficient (CC)**: Between true vs predicted signals (alternative to R²).
   - **Mean Squared Error (MSE)** or **RMSE** for absolute signal differences.

3. **Trial-based data:**
   Models handle discontinuous segments (e.g., trials with varying durations). Each trial is independently processed, and outputs are concatenated for evaluation.

---

### Data for Training, Validation, Testing:
- **Training:** Time-series input (`y`, `z`, `u`), optionally mean-removed or z-scored during pre-normalization.
- **Validation:** Held-out portion of training data. Often inner folds or separate sets help tune hyperparameters.
- **Testing:** Independent datasets, distinct from training and validation. These are fed to the `predict()` method for latent-state estimation and decoding.

---

### Cross-validation Feasibility
Yes, cross-validation is fully feasible. Standard procedures apply:
   - Use inner CV within training for hyperparameter tuning (e.g., via grid search).
   - Outer test validation remains exclusive for unbiased performance evaluation.

With IPSID's focus on minimal state learning specific to the experimental goal, CV often involves:
   - Searching for `nx`, `n1`, and horizon `i`.
   - High `i` helps small-dimension signals (`y`, `z`), while low `i` avoids unnecessary complexity for rich data.

---

If you aim to proceed practically, let me know, and I'll help set up an example!


### Hyperparameters in PSID/IPSID Training

To optimize and adapt PSID/IPSID to specific datasets, certain parameters serve as hyperparameters. These parameters influence the model complexity, training process, and performance:

1. **Model-Specific Hyperparameters:**
   - **`n_states` (or `nx`)**: Total dimensionality of the latent state space. Determines how many latent variables are used to model the dynamics.
   - **`rank_n`**: Limits the rank of the state subspace (optional in some configurations).
   - **`past_horizon` and `future_horizon` (`i` in PSID)**: Block-row size for past and future segments in Hankel matrices. This parameter effectively determines the temporal window used for learning.
   - **`n1`**: The number of behaviorally relevant latent states (only applicable if behavior is included).
   - **`alpha`**: Regularization strength for least-squares fitting. Useful for controlling overfitting during matrix decomposition.
   - **`stable_A`**: Whether the system enforces stability in the state dynamics (ensuring all eigenvalues of `A` are within unit circle for discrete systems).
   - **`estimate_noise`**: Boolean flag to determine if noise parameters (`Q`, `R`, `S`) should be estimated.

2. **Training-Specific Hyperparameters:**
   - **Learning rate** (`training.learning_rate`): Determines the size of parameter updates during optimization.
   - **Weight decay** (`training.weight_decay`): Regularization parameter to avoid overfitting by penalizing large weights.
   - **Batch size** (`data.batch_size`): Number of samples per training batch. Impacts memory consumption and gradient estimation.
   - **Gradient clipping** (`training.gradient_clip_norm`): Applies clipping to gradients to prevent exploding gradients during backpropagation.
   - **Scheduler decay rate and step size:** Adaptively lowers the learning rate throughout training.

3. **Data-Specific Hyperparameters:**
   - **Data preprocessing:** Whether to standardize (z-score) or zero-mean the data (e.g., `data.standardize.neural` or `data.standardize.behavior`).
   - **Split settings:** Partitioning the data into training/validation/test sets (validation size, test size, and random seeds).

4. **Early Stopping Criteria:**
   - **Metric for early stopping** (`early_stopping.metric`): Typically monitors validation loss or a performance metric like R².
   - **Patience** (`early_stopping.patience`): Number of epochs of no improvement before terminating training.

---

### Methods Used for Full PSID/IPSID Training Workflow

To manage the full workflow of training, validation, and testing, methods would typically include the following steps:

#### 1. **Preprocessing:**
   - **Mean Removal & Standardization:** Ensure all input data (`y`, `z`, `u`) is zero-mean or z-scored. This can be supported by preprocessors like `PrepModel`.

#### 2. **Hankel Matrix Construction:**
   - Use `past_horizon` and `future_horizon` to construct block Hankel matrices for historical and future sequences. These matrices form the basis for identifying the state-space model.

#### 3. **Parameter Estimation:**
   - **Stage 1 (Behaviorally Relevant States):**
     - Learning the subspace (`x1`) that prioritizes encoding `z` using projections between longitudinal components of `y` and `z`.
     - Fit parameters like system dynamics matrices (`A`, `Cz`).
   - **Stage 2 (Additional State Dimensions):**
     - Fit remaining latent states (`x2`) that model intrinsic neural dynamics unrelated to behavior.
   - **Optional IPSID Step (Input Differentiation):**
     - Further decouple dynamics due to external inputs (`u`) using preprocessing with `x3`.

#### 4. **Training and Validation Loops:**
   - **Training:** Update weights for state-space matrices using least-squares or optimization procedures. For IPSID, include input-driven dissociation.
   - **Validation:** Evaluate metrics like R² and loss (`val_loss`) on a validation set. Use validation results to monitor overfitting and guide hyperparameter tuning.

#### 5. **Hyperparameter Tuning:**
   - Grid or random search over hyperparameters like `n_states`, `n1`, `past_horizon`, `future_horizon`, learning rate, and regularization parameters based on validation performance.
   - Incorporate constraints during search (e.g., `rank_n` ≤ `n_states`).

#### 6. **Testing:**
   - Apply the trained model to an independent test dataset, evaluating behavioral decoding and neural activity prediction accuracy (e.g., using R² metrics).

---

### Example Pseudocode for Complete Workflow

Below is a simplified skeleton for implementing the training workflow:

```python
class PSIDTrainer:
    def __init__(self, config, model_type="PSID"):
        self.config = config
        self.model_type = model_type
        self.data_preprocessor = None
        self.model = None

    def preprocess_data(self, Y, Z=None, U=None):
        # Mean or z-score inputs based on config
        self.data_preprocessor = PrepModel()
        self.data_preprocessor.fit(Y, zscore=self.config.data.standardize.get('neural', False))
        Y = self.data_preprocessor.apply(Y)
        if Z is not None:
            Z = self.data_preprocessor.apply(Z)
        if U is not None:
            U = self.data_preprocessor.apply(U)
        return Y, Z, U

    def initialize_model(self):
        model_cfg = self.config.model
        if self.model_type == "PSID":
            return PSID(
                n_states=model_cfg.n_states,
                n1=model_cfg.get("n1"),
                past_horizon=model_cfg.past_horizon,
                alpha=model_cfg.alpha,
                stable_A=model_cfg.stable_A
            )
        elif self.model_type == "IPSID":
            return IPSID(...)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def train(self, Y_train, Z_train, U_train=None, Y_val=None, Z_val=None, U_val=None):
        Y_train, Z_train, U_train = self.preprocess_data(Y_train, Z_train, U_train)
        self.model = self.initialize_model()
        
        val_loss_epochs = []
        for epoch in range(self.config.training.epochs):
            # Perform a training step
            self.model.train(Y_train, Z_train, U_train)
            
            # Validation step
            if Y_val is not None:
                val_loss = self.model.validate(Y_val, Z_val, U_val)
                val_loss_epochs.append(val_loss)
                print(f"Epoch {epoch}, Validation Loss: {val_loss}")
            
            # Early stopping
            early_stop = (
                self.config.training.early_stopping.enabled and
                len(val_loss_epochs) > self.config.training.early_stopping.patience and
                min(val_loss_epochs[-self.config.training.early_stopping.patience:]) >= min(val_loss_epochs)
            )
            if early_stop:
                print(f"Stopping early at epoch {epoch}")
                break

    def test(self, Y_test, Z_test, U_test=None):
        Y_test, Z_test, U_test = self.preprocess_data(Y_test, Z_test, U_test)
        results = self.model.test(Y_test, Z_test, U_test)
        print("Test metrics:", results)

# Example usage
config = load_config('ndm_psid.yaml')  # Load provided YAML
trainer = PSIDTrainer(config, model_type="PSID")
trainer.train(Y_train, Z_train, U_train, Y_val, Z_val, U_val)
trainer.test(Y_test, Z_test, U_test)
```


---

This workflow demonstrates explicit control over preprocessing, training, validation, early stopping, and testing. Most hyperparameters can be dynamically tuned via grid or random search using validation set performance. Let me know if you'd like a more detailed implementation for a particular step!


### Setting Up Neural-only vs. Neural + Behavioral PSID

You can configure PSID to solve a neural-only task (self-prediction or dynamics modeling for neural activity) or a neural + behavioral task (e.g., modeling behaviorally relevant neural dynamics). Below are the key configurations and methods.

---

#### 1. **Neural-only PSID**

For a neural-only model, you only provide the neural data (`y`) for both inputs and outputs (self-prediction):
- **Input data (`y`)**: Neural time-series data `Y` is passed in as both independent variables and the targets.
- **Skip behavioral signal (`z`)**: Set `Z=None` (no behavioral data).

##### Sample Initialization

```python
import PSID
# Initialize PSID
psid_neural = PSID.PSID(
    Y=neural_data,    # Neural data as the input
    Z=None,           # No behavioral data
    nx=20,            # Total latent states
    n1=0,             # No behavioral subspace, just self-prediction of `y`
    i=10,             # Past and future horizon block size
)
```


##### Core Changes for Neural-only:
- Skip `Z`-related configs:
  - `n1=0`: No behaviorally relevant states are extracted. All `nx` is for neural dynamics.
- Preprocess `Y` (neural signals) using mean removal or z-scoring:
  - Use `remove_mean_Y=True` or `zscore_Y=True`.

---

#### 2. **Neural + Behavioral PSID**

For models incorporating behavioral dynamics, provide both neural signals (`y`) and behavioral outputs (`z`):
- **Input data (`y`)**: Neural time-series data.
- **Behavioral signal (`z`)**: Provide behavioral outputs as `Z` for decoding.
- Configure `n1`: Set the number of behaviorally relevant latent dimensions.

##### Sample Initialization

```python
psid_neural_behavioral = PSID.PSID(
    Y=neural_data,    # Neural data as the input
    Z=behavior_data,  # Behavior output
    nx=20,            # Total latent states
    n1=10,            # Split latent states (10 for behavior-relevant, 10 for others)
    i=10,             # Past and future horizon block size
)
```


##### Core Changes for Neural + Behavioral:
- **Behavioral Latent States (`n1`)**: Defines how many latent states specifically encode `Z`-to-`Y` relationships.
- **`Y` and `Z` Alignment**: Ensure both neural and behavioral data are aligned with respect to time.

---

#### 3. **Concatenating or Splitting Input Channels**

To configure input channels (`Y` or multi-modal data like `Y` and `Z`) flexibly, you can choose whether to concatenate neural and behavioral data or treat them separately:

- **Concatenation (`concat`)**: Combine `Y` (neural) and `Z` (behavioral) into a single matrix for training.  
- **Separate Channels**: Pass them as distinct arguments during initialization. Behavioral subspaces (`x1`) and intrinsic ones (`x2`) are handled differently.

##### Example: Separate Channels

```python
# Provide neural and behavioral data separately
psid = PSID.PSID(
    Y=neural_data,    # Neural signals
    Z=behavior_data,  # Behavior
    nx=24,            # Total latent state dimensionality
    n1=12,            # Half of states dedicated to behaviorally relevant signals
    i=10,             # Temporal horizon
)
```


##### Example: Concatenated Inputs

If concatenating all signals into a single input matrix:
```python
import numpy as np

# Concatenate neural and behavioral signals as a single input matrix
input_data = np.concatenate((neural_data, behavior_data), axis=1)
psid = PSID.PSID(
    Y=input_data,    # Combined dataset
    Z=behavior_data, # `Z` still needed as the decoding target 
    nx=30,           # Larger latent state dimension
    n1=15,           # States split to encode decoding capacity for `Z`
    i=15,            # Increased time horizon
)
```


Set the `n1` proportional to the behavioral signal's influence, while the rest of `nx` models shared or intrinsic neural dynamics.

---

#### 4. **Retrieving All Matrices**

After training, the learned system matrices are accessible via the PSID model object:

- **Dynamic Matrix (`A`)**: Transition matrix for latent states.
- **Input Matrix (`B`)**: Influence of external input (if present) on states.
- **Output Matrices (`C_y`, `C_z`)**: Map states to neural (`C_y`) and behavioral outputs (`C_z`).
- **Noise Covariance (`Q`, `R`, `S`)**: Process and observation noise.

##### Retrieving Model Parameters

```python
# Get parameters after training
model = psid.train(Y_train, Z_train)
A = model.A      # Latent dynamics matrix
B = model.B      # Input coupling (external input dynamics)
C_y = model.C    # Neural output transformation (states to `y`)
C_z = model.Cz   # Behavioral output transformation (states to `z`)
Q = model.Q      # Process noise covariance
R = model.R      # Observation noise covariance
S = model.S      # Cross-covariance noise terms
```


If you're especially interested in components driving the behavior (`C_z`):
```python
print("Behavioral Decoding Matrix (C_z):", model.Cz)
```


---

#### 5. **Configuring Extra Behavioral/Neural Layers**

If you’re using decoders or encoders for input channels, models like PSID and IPSID allow for various combinations of neural and behavioral inputs:

##### Neural Encoder (`Y` Only -> Latent Space)

```python
# Neural encoder
latent_states = encoder(neural_data)  # Apply mapping Y -> x(k)
```


##### Fusion: Neural + Behavioral

```python
# Concatenate neural and behavioral channels
fusion_input = np.concatenate([neural_features, behavioral_features], axis=-1)

# Apply fusion mechanism (e.g., MLP)
fused_latent = fusion_model(fusion_input)  # Fusion into latent space
```


---

Let me know if you'd like details on setting data preprocessing pipelines, interpretation of matrix outputs, or advanced training steps.