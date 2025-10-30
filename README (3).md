# 🏦 Loan Approval Optimization — Deep Learning & Offline Reinforcement Learning

This repository provides a complete end-to-end pipeline for **loan approval optimization**, integrating both **Deep Learning (DL)** and **Offline Reinforcement Learning (RL)** techniques.

The primary objective is to help a fintech institution **maximize profit** while **minimizing loan defaults** by learning from historical data and optimizing its approval strategy.

---

## 🧭 Project Overview

The workflow is organized into four major tasks:

1. **Task 1 — Exploratory Data Analysis (EDA) & Preprocessing**
   Data cleaning, feature engineering, and dataset preparation.
2. **Task 2 — Deep Learning Model (PyTorch MLP)**
   Supervised learning model to predict default probability.
3. **Task 3 — Offline Reinforcement Learning (CQL)**
   RL-based agent that learns profit-maximizing loan approval policies using offline data.
4. **Task 4 — Comparative Analysis & Insights**
   Evaluation of DL (AUC/F1) vs. RL (Estimated Policy Value), policy comparison, and future recommendations.

All experiments can be reproduced locally (Python **3.11** recommended) or on **Google Colab**.

---

## 🗁 Folder Structure

```
.
├── data/
│   ├── accepted_2007_to_2018Q4.csv.gz      # Raw dataset (optional)
│   └── loan_clean_subset.csv                # Cleaned dataset (from Task 1)
│
├── notebooks/
│   ├── Task1_EDA_Preprocess.ipynb
│   ├── Task2_DL_Predictive_Model.ipynb
│   ├── Task3_Offline_RL_CQL.ipynb
│   └── Task4_Analysis_Comparison.ipynb      # Combined analysis notebook
│
├── models/
│   ├── model_mlp_default_risk/
│   │   └── pytorch_mlp.pt                   # Saved DL model weights
│   └── offline_rl_cql/
│       ├── cql_discrete_model.d3            # Trained RL policy
│       └── preprocess.joblib                # Preprocessor for inference
│
├── requirements.txt
└── README.md
```

> 💡 **Note:** If the raw dataset is unavailable, you can still run the pipeline using `data/loan_clean_subset.csv` generated during Task 1.

---

## ⚙️ Environment Setup

### 🧬 Option A — Conda (Recommended)

> ⚠️ Use **Python 3.11** — newer versions (e.g., 3.13) may break compatibility with RL and TensorFlow dependencies.

```bash
# From the repository root
conda create -n loan-rl python=3.11 -y
conda activate loan-rl

# Install dependencies
python -m pip install -U pip
pip install -r requirements.txt

# (Optional) Add Jupyter kernel
python -m ipykernel install --user --name loan-rl --display-name "Python 3.11 (loan-rl)"

# Launch Jupyter Notebook
jupyter notebook
# Then in Jupyter: Kernel → Change Kernel → "Python 3.11 (loan-rl)"
```

---

## 🗃️ Data Files

Place the following files inside the **`data/`** directory:

* **Preferred:** `loan_clean_subset.csv`
  – Cleaned dataset generated after preprocessing
  – Includes features and target column (`0 = Fully Paid`, `1 = Defaulted`)

* **Optional:** `accepted_2007_to_2018Q4.csv.gz`
  – Raw dataset; Task 1 will automatically generate `loan_clean_subset.csv`

> Ensure that the columns **`loan_amnt`** and **`int_rate`** are present — both are essential for RL reward computation.

---

## 🚀 How to Run the Pipeline

### 🔹 Step 1 — EDA & Preprocessing

Open **`notebooks/Task1_EDA_Preprocess.ipynb`** → **Run All**
This notebook:

* Loads raw data (if available)
* Cleans and preprocesses features
* Saves `data/loan_clean_subset.csv`

**Output:** `data/loan_clean_subset.csv`

---

### 🔹 Step 2 — Deep Learning Model (DL)

Open **`notebooks/Task2_DL_Predictive_Model.ipynb`** → **Run All**
This notebook:

* Loads the cleaned dataset
* Builds preprocessing and training pipelines (imputation + OHE + scaling)
* Trains a **PyTorch MLP** for default prediction
* Evaluates using **AUC** and **F1-Score**

**Outputs:**

* `models/model_mlp_default_risk/pytorch_mlp.pt`
* Printed metrics: AUC, F1, classification report, confusion matrix

---

### 🔹 Step 3 — Offline Reinforcement Learning (CQL)

Open **`notebooks/Task3_Offline_RL_CQL.ipynb`** → **Run All**
This notebook:

* Converts data into a **bandit-style MDP** (actions = {approve, deny})
* Trains a **Conservative Q-Learning (CQL)** agent using **d3rlpy**
* Computes **Estimated Policy Value (EPV)** on the test set

**Outputs:**

* `models/offline_rl_cql/cql_discrete_model.d3`
* `models/offline_rl_cql/preprocess.joblib`
* Printed metrics: EPV, approval rate, approval breakdown

---

### 🔹 Step 4 — Analysis & Comparison

Open **`notebooks/Task4_Analysis_Comparison.ipynb`** → **Run All**
This notebook:

* Reevaluates DL metrics (AUC, F1)
* Computes RL EPV and (optionally) FQE
* Displays cases where DL and RL disagree
* Generates the **Final Summary Report**

**Outputs:**

* Metrics summary (AUC, F1, EPV, approval rate)
* Policy disagreement table
* Final one-page report

---

## 💻 Command-Line Quickstart (Optional)

Run all major tasks in one go:

```bash
# Ensure cleaned dataset exists
python notebooks/Task4_Analysis_Comparison.ipynb
```

This script will:

* Train and evaluate the DL model (AUC/F1)
* Train and evaluate the RL agent (EPV)
* Show policy disagreements and final results

---

## 🤖 Troubleshooting Guide

| **Issue**                                            | **Solution**                                                               |
| ---------------------------------------------------- | -------------------------------------------------------------------------- |
| TensorFlow/d3rlpy incompatibility (Python 3.13)      | Use **Python 3.11** as shown in setup instructions.                        |
| `ModuleNotFoundError: gymnasium.wrappers.time_limit` | Install `gymnasium[classic-control]==0.29.1`.                              |
| RL fit() signature mismatch                          | Use positional form: `algo.fit(train_dataset, 100_000)` (already applied). |
| GPU not detected                                     | CPU works fine; ensure CUDA + PyTorch GPU build for acceleration.          |
| Missing `loan_amnt` or `int_rate` columns            | Re-run Task 1 to regenerate the cleaned dataset.                           |

---

## 📊 Expected Outputs

| **Model Type**                   | **Key Metrics**              | **Expected Outputs**                                      |
| -------------------------------- | ---------------------------- | --------------------------------------------------------- |
| **Deep Learning (MLP)**          | ROC-AUC, F1                  | Classification Report, Confusion Matrix                   |
| **Reinforcement Learning (CQL)** | Estimated Policy Value (EPV) | Approval Rate, Approved-Paid vs Approved-Defaulted Counts |
| **Comparison Report**            | AUC/F1 vs EPV                | Policy Disagreement Cases + Summary Analysis              |

---

## 🧠 Libraries & References

* **PyTorch** — Deep Learning framework
* **scikit-learn** — Preprocessing, metrics, data splitting
* **d3rlpy** — Offline RL (Conservative Q-Learning, FQE)
* **gymnasium** — RL environment dependencies

---

## 📈 Results Summary

| **Model**           | **Metric**                 | **Score**        |
| ------------------- | -------------------------- | ---------------- |
| Deep Learning (MLP) | **F1-Score**               | **0.747**        |
| Deep Learning (MLP) | **AUC**                    | **0.928**        |
| Offline RL (CQL)**  | **Estimated Policy Value** | Profit-Optimized |

These results confirm that the **DL model** effectively predicts credit risk, while the **RL agent** learns a more **reward-oriented policy** balancing profit and risk.

---

## 🗾 Requirements

```bash
numpy<2.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
joblib>=1.3.0

# Deep Learning (PyTorch)
torch==2.4.1

# Offline RL (CQL) and dependencies
d3rlpy==2.4.0
gymnasium[classic-control]==0.29.1

# Optional: Jupyter support
ipykernel>=6.29.0
```

---

## ⚖️ License

This project is intended **solely for educational and internal evaluation** purposes.
Use responsibly and cite appropriately if adapted for research or academic work.
