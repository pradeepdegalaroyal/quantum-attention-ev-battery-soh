# MQAttn-QLSTM-VQR

**Quantum Recurrent Attention for Causal Electric-Vehicle Battery State-of-Health Prognosis**

Code, data, trained models and results for the manuscript *"Quantum Recurrent Attention for Causal Electric-Vehicle Battery State-of-Health Prognosis: A Parameter-Matched, Hardware-Aware Evaluation"*.

## Overview

MQAttn-QLSTM-VQR forecasts the state of health (SOH) of electric-vehicle batteries from charging data. The name stands for multi-qubit quantum attention, quantum long short-term memory and variational quantum residual. The model has 988 parameters and three parts:

1. a quantum long short-term memory (QLSTM) cell that reads a six-month window of charging features;
2. a quantum attention layer whose query, key and value maps are variational quantum circuits (the similarity and the softmax are classical);
3. a variational quantum residual head that corrects the primary output with the last month's charging statistics.

The model predicts the change of SOH from the last observed value and uses a backward-only (causal) target. Each quantum model has a *classical twin* in which every circuit is replaced by a classical layer of the same size, so that the effect of the circuits can be measured.

## Key results

Mean absolute error (MAE, % of SOH) on the vehicle data, causal target, one-month horizon, mean of two seeds and ten leave-vehicle-out folds:

| Model | Parameters | MAE (%) |
|---|---:|---:|
| MQAttn-QLSTM-VQR | 988 | 0.399 |
| Classical twin of MQAttn-QLSTM-VQR | 988 | 0.394 |
| Seq2Seq | 33,345 | 0.384 |
| Persistence | 0 | 0.553 |

- The model beats persistence on every held-out vehicle and cell of three datasets (BatICM, NASA, Oxford), with a skill between 17% and 61%.
- Its error shows no significant difference to that of a Seq2Seq network with 34 times more parameters, and its mean error is the lowest when only four vehicles train it.
- Its classical twin shows no significant difference in any setting, so the results attribute the benefit to the compact design and not to the circuits.
- Training with parameter-shift gradients and 1,024 shots gives the same error as exact simulation.

All tables and figures of the manuscript are generated from the files in `results/`.

## Repository structure

```
qsoh/                  Python package
    models.py          MQAttn-QLSTM-VQR, QLSTM, LSTM, GRU, MLP, Seq2Seq
    simulator.py       exact batched circuit simulator, noise model, parameter-shift gradients
    data.py            preprocessing of the BatICM vehicle data
    cell_datasets.py   preprocessing of the NASA and Oxford cell data
experiments/           command-line scripts
    run_comparison.py  main comparison of all models
    run_ablation.py    ablation study
    run_hardware.py    shots, gate noise, readout error and parameter-shift training
    run_cost.py        parameters, memory, FLOPs and circuit counts
    train_checkpoints.py
    make_tables_figures.py
results/               predictions for every window, summaries and statistical tests
checkpoints/           trained MQAttn-QLSTM-VQR for each of the ten folds
data/processed/        monthly features of the 20 BatICM vehicles
```

Result files follow the pattern `<dataset>_<target>_h<horizon>_q<qubits>l<layers>_ntr<training vehicles>_s<seed>_<content>.csv`.

## Installation

```bash
git clone https://github.com/pradeepdegalaroyal/quantum-attention-ev-battery-soh.git
cd quantum-attention-ev-battery-soh
pip install -r requirements.txt
```

Python 3.11 is required. All experiments run on a CPU with one thread per process; no GPU is needed.

## Data

| Dataset | Source | Location |
|---|---|---|
| BatICM, 20 on-road electric vehicles (Deng et al., *Applied Energy* 339, 2023) | <https://github.com/BatICM/battery-charging-data-of-on-road-electric-vehicles> (MIT licence) | `data/BatICM/vehicles/` |
| NASA battery dataset (Saha and Goebel, 2007) | <https://data.nasa.gov/dataset/li-ion-battery-aging-datasets> | `data/external/nasa_pcoe/cleaned_dataset/` |
| Oxford Battery Degradation Dataset 1 (Birkl, 2017) | doi:10.5287/bodleian:KO2kdmYGg | `data/external/oxford_degradation_1/` |

The processed BatICM features are included (`data/processed/baticm_monthly.pkl`). The raw BatICM, NASA and Oxford files are needed only to rebuild the processed data.

## Usage

Run all commands from the repository root.

```bash
# Preprocessing (optional)
python -m qsoh.data
python -m qsoh.cell_datasets

# Main comparison
python -m experiments.run_comparison --target causal --seed 0
python -m experiments.run_comparison --target causal --seed 1
python -m experiments.run_comparison --target causal --horizon 3
python -m experiments.run_comparison --target causal --horizon 6
python -m experiments.run_comparison --target raw
python -m experiments.run_comparison --target lowess_full
python -m experiments.run_comparison --target causal --n_train_veh 4
python -m experiments.run_comparison --dataset nasa --horizon 1
python -m experiments.run_comparison --dataset nasa --horizon 10
python -m experiments.run_comparison --dataset oxford --horizon 1
python -m experiments.run_comparison --dataset oxford --horizon 5

# Ablation, one run per seed (seeds 0 to 4)
python -m experiments.run_ablation --target causal --lrs 1e-2 --seed 0

# Number of qubits and circuit depth
python -m experiments.run_comparison --models MQAttn-QLSTM-VQR --nq 2 --lv 10
python -m experiments.run_comparison --models MQAttn-QLSTM-VQR --nq 3 --lv 8
python -m experiments.run_comparison --models MQAttn-QLSTM-VQR --nq 4 --lv 5

# Hardware realism, one run per fold (folds 1 to 10)
python -m experiments.run_hardware --fold 1

# Cost of a prediction
python -m experiments.run_cost

# Tables and figures
python -m experiments.make_tables_figures --out outputs
```

## Trained models

`checkpoints/mqattn_qlstm_vqr_fold{1..10}.pt` hold the trained weights, the input scaler, the residual offset and the test error of each fold.

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

## Verification

- `qsoh/simulator.py` agrees with PennyLane `default.qubit` for (qubits, layers) in {(4,2), (3,8), (2,10), (4,5), (6,2)}: outputs to within 4·10⁻⁷ and gradients to within 4·10⁻⁶.
- Its parameter-shift gradients agree with back-propagation on the full model to a relative error below 4·10⁻⁹.
- Each checkpoint reproduces the test error reported for its fold.

## Citation

If you use this software, the trained models or the results, please cite it. GitHub shows a ready-made citation under **Cite this repository** (from `CITATION.cff`).

```bibtex
@misc{chenchupradeep2026code,
  author       = {Chenchupradeep, Degala},
  title        = {{MQAttn-QLSTM-VQR}: code, data, trained models and results for causal
                  electric-vehicle battery state-of-health prognosis},
  year         = {2026},
  howpublished = {GitHub repository},
  note         = {Version 1.0},
  url          = {https://github.com/pradeepdegalaroyal/quantum-attention-ev-battery-soh}
}
```

Please also cite the data sources listed above.

## Licence

Copyright © 2026 Degala Chenchupradeep. All rights reserved. This material may be viewed and run only to review, verify or reproduce the results of the associated manuscript. Any other use requires written permission. See [LICENSE](LICENSE).

## Contact

Degala Chenchupradeep, SRM University-AP, Andhra Pradesh, India
chenchupradeep_d@srmap.edu.in · ORCID [0009-0003-6959-6295](https://orcid.org/0009-0003-6959-6295)
