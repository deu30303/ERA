ERA: Evidence-based Reliability Alignment for Honest Retrieval-Augmented Generation

This repository contains the official implementation of the paper **"ERA: Evidence-based Reliability Alignment for Honest Retrieval-Augmented Generation"**.

ERA is a principled reliability alignment framework designed to address critical knowledge conflicts in Retrieval-Augmented Generation (RAG) systems. By leveraging Evidential Deep Learning and Dempster-Shafer Theory, ERA shifts confidence estimation from scalar probabilities to explicit evidence distributions. This geometric quantification ensures that the model effectively disentangles epistemic uncertainty from data ambiguity, allowing it to consistently reject unsupported queries and maintain robust calibration even under noisy retrieval scenarios.

## 🌟 Key Features

- **Evidence-based Confidence Estimation:** Shifts confidence quantification from scalar probabilities to explicit evidence distributions via Evidential Deep Learning (EDL), effectively disentangling epistemic uncertainty from aleatoric ambiguity.
- **Geometric Conflict Quantification:** Leverages Dempster-Shafer Theory (DST) to rigorously measure the geometric discordance between internal parametric knowledge and external retrieved evidence, providing a precise "Conflict Score".
- **Conflict-Aware Alignment:** Implements a dynamic optimization strategy that modulates the Direct Preference Optimization (DPO) objective based on detected conflicts, ensuring the model prioritizes honest abstention over hallucination in high-uncertainty scenarios.

## License
ERA and its family are released under the CC BY-NC 4.0 License.

## 📂 Repository Structure

```bash
.
├── dpo_trainer_era.py      # [Core] Implementation of ERA using Direct Preference Optimization (DPO)
├── evaluations.py          # [Core] Evaluation script for RAG performance and honesty metrics
├── preprocess/             # Data preprocessing modules
│   ├── build_dataset.py    # Script for constructing alignment datasets
│   └── process_data.py     # Utility for cleaning and formatting raw data
├── tuner/                  # Model tuning and training utilities
│   ├── train_utils.py      # Helper functions for the training loop
│   └── loss_functions.py   # Implementation of Evidence-based loss (EDL)
├── metrics/                # Implementation of geometric conflict and reliability metrics
│   ├── conflict_score.py   # Calculates conflict between parametric and retrieved knowledge
│   └── uncertainty.py      # Quantifies epistemic uncertainty using Dempster-Shafer Theory
├── data_kbrag/             # Directory containing processed datasets (KBRAG, etc.)
├── result/                 # Directory for storing trained models and evaluation outputs
└── README.md               # Project documentation and setup guide
