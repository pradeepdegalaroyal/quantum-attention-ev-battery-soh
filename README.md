# Quantum Recurrent Attention for Causal EV Battery State-of-Health Prognosis

Code, processed data, trained models and results for the paper

> D. Chenchupradeep, *"Quantum Recurrent Attention for Causal Electric-Vehicle Battery State-of-Health Prognosis: A Parameter-Matched, Hardware-Aware Evaluation,"* submitted to the IEEE Open Journal of Vehicular Technology.

**MQAttn-QLSTM-VQR** (multi-qubit quantum attention, quantum long short-term memory and variational quantum residual) forecasts the state of health (SOH) of electric-vehicle batteries. It has 988 parameters and three parts:
- a quantum long short-term memory (QLSTM) cell;
- an attention layer whose query, key and value maps are variational quantum circuits;
- a variational quantum residual head.

The model predicts the change of SOH from the last observed value, using a backward-only (causal) target.

Every number, table and figure of the paper can be regenerated from this repository.

## Licence

© 2026 Degala Chenchupradeep. All rights reserved. You may view and run this material **only to review, verify or reproduce the results of the paper**. Any other use requires written permission; see [LICENSE](LICENSE). Please cite the paper ([CITATION.cff](CITATION.cff)). Release `v1.0` is the version that accompanies the paper.

## Repository layout

| Path | Content |
|---|---|
| `qsoh/models.py` | MQAttn-QLSTM-VQR, QLSTM, LSTM, GRU, MLP and Seq2Seq models |
| `qsoh/simulator.py` | Exact batched simulator of the quantum circuits, noise model and parameter-shift gradients |
| `qsoh/data.py` | Preprocessing of the BatICM vehicle data |
| `qsoh/cell_datasets.py` | Preprocessing of the NASA and Oxford cell data |
| `experiments/run_comparison.py` | Main comparison of all models (Tables 1, 2 and 4) |
| `experiments/run_ablation.py` | Ablation of MQAttn-QLSTM-VQR (Table 3) |
| `experiments/run_hardware.py` | Shots, gate noise, readout error and parameter-shift training (Table 5) |
| `experiments/run_cost.py` | Parameters, memory, FLOPs and circuit counts (Table 6) |
| `experiments/train_checkpoints.py` | Trains and saves MQAttn-QLSTM-VQR for each fold |
| `experiments/make_tables_figures.py` | Builds every table and figure from `results/` |
| `results/` | Predictions for every window, summaries and statistical tests |
| `checkpoints/` | Trained MQAttn-QLSTM-VQR for each of the 10 folds |
| `data/processed/baticm_monthly.pkl` | Monthly features of the 20 BatICM vehicles |

Result files follow the pattern `<dataset>_<target>_h<horizon>_q<qubits>l<layers>_ntr<training vehicles>_s<seed>_<content>.csv`:
- `ntrall` means all training vehicles;
- `_ru` marks data re-uploading;
- `_ablation` marks the ablation runs.

## Data

| Dataset | Source | Location |
|---|---|---|
| BatICM, 20 on-road EVs (Deng et al., *Applied Energy* 339, 2023) | <https://github.com/BatICM/battery-charging-data-of-on-road-electric-vehicles> (MIT licence) | `data/BatICM/vehicles/` (needed only to rebuild `data/processed/`) |
| NASA battery dataset (Saha and Goebel, 2007) | <https://data.nasa.gov/dataset/li-ion-battery-aging-datasets> | `data/external/nasa_pcoe/cleaned_dataset/` |
| Oxford Battery Degradation Dataset 1 (Birkl, 2017) | doi:10.5287/bodleian:KO2kdmYGg | `data/external/oxford_degradation_1/Oxford_Battery_Degradation_Dataset_1.mat` |

## Installation

Python 3.11:

```bash
pip install -r requirements.txt
```

All experiments run on a CPU with one thread per process; a GPU is not required.

## Reproducing the results

Run the commands from the repository root.

```bash
# Data (optional: data/processed/baticm_monthly.pkl is included)
python -m qsoh.data
python -m qsoh.cell_datasets

# Main comparison (Table 2, Figs. 5, 6 and 8)
python -m experiments.run_comparison --target causal --seed 0
python -m experiments.run_comparison --target causal --seed 1
python -m experiments.run_comparison --target causal --horizon 3
python -m experiments.run_comparison --target causal --horizon 6
python -m experiments.run_comparison --target raw
python -m experiments.run_comparison --target causal --n_train_veh 4
python -m experiments.run_comparison --dataset nasa --horizon 1
python -m experiments.run_comparison --dataset nasa --horizon 10
python -m experiments.run_comparison --dataset oxford --horizon 1
python -m experiments.run_comparison --dataset oxford --horizon 5

# Effect of the target (Table 1)
python -m experiments.run_comparison --target lowess_full

# Ablation (Table 3), one run per seed
python -m experiments.run_ablation --target causal --lrs 1e-2 --seed 0   # repeat for seeds 1-4

# Number of qubits and circuit depth (Table 4)
python -m experiments.run_comparison --models MQAttn-QLSTM-VQR --nq 2 --lv 10
python -m experiments.run_comparison --models MQAttn-QLSTM-VQR --nq 3 --lv 8
python -m experiments.run_comparison --models MQAttn-QLSTM-VQR --nq 4 --lv 5

# Hardware realism (Table 5, Fig. 7), one run per fold
python -m experiments.run_hardware --fold 1   # repeat for folds 2-10

# Cost of a prediction (Table 6)
python -m experiments.run_cost

# All tables and figures -> outputs/
python -m experiments.make_tables_figures --out outputs
```

## Verification

`qsoh/simulator.py` agrees with PennyLane `default.qubit` for (qubits, layers) ∈ {(4,2), (3,8), (2,10), (4,5), (6,2)}:
- outputs agree to within 4·10⁻⁷;
- gradients agree to within 4·10⁻⁶;
- its parameter-shift gradients agree with back-propagation on the full model to a relative error below 4·10⁻⁹.

Each checkpoint in `checkpoints/` reproduces the test error that the paper reports for its fold.

## Using a trained model

```python
import torch
import qsoh.models as M
from qsoh import simulator

simulator.use_fast_backend(M)
ck = torch.load("checkpoints/mqattn_qlstm_vqr_fold1.pt", weights_only=False)
model = M.MQAttn_QLSTM_VQR(**ck["config"])
model.load_state_dict(ck["state_dict"])
model.eval()
# x: (batch, 6, 11) window of the 10 monthly features and the causal SOH,
#    standardised with ck["scaler_mean"] and ck["scaler_scale"]
# prediction = last observed SOH + ck["residual_offset_mu"] + model(x)
```

## Citation

Please cite the paper (see `CITATION.cff`) and the three dataset sources above.
