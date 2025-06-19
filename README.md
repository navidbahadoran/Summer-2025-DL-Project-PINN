# PINN-based COVID-19 Modeling using NYT Data

This project demonstrates how to use **Physics-Informed Neural Networks (PINNs)** to model the spatiotemporal spread of COVID-19 using real-world data from the **New York Times (NYT)** COVID-19 dataset. It includes data preprocessing, training of a PINN model, comparison with traditional solvers, and exploratory data analysis (EDA).

---

## 📁 Project Structure

```
pinn_project/
│
├── checkpoints/              # 🔒 Saved model weights (e.g., model.pth)
│
├── models/
│   └── neural_net.py         # 🧠 PINN network architecture
│
├── data/
│   ├── covid_county_cases.csv  # ✅ Processed NYT dataset
│   └── data_generator.py       # 🔁 Loads/normalizes data and creates training/test sets
│
├── pinn/
│   └── pinn_solver.py        # 🔍 PINN residual formulation, autograd, loss
│
├── traditional/
│   └── fd_solver.py          # 🧮 Finite Difference solver for benchmarking (optional)
│
├── scripts/
│   └── nyt_to_csv.py         # 🗂️ Converts NYT + shapefile + population into usable CSV
│
├── main_train.py             # 🚀 Main training script for PINN
├── evaluate.ipynb            # 📊 Model evaluation and visualization
├── EDA.ipynb                 # 📈 Exploratory data analysis
├── config.py                 # ⚙️ All hyperparameters and paths
└── utils.py                  # 📦 Helper functions: RMSE, plotting, etc.
```

---

## 📌 Dataset Overview

We use:

- **NYT COVID-19 US county-level dataset**
- **US Census shapefiles** for geometry
- **County-level population estimates** for normalization

The preprocessing script `scripts/nyt_to_csv.py` merges all of these and produces a CSV:

```
lon, lat, t, u
```

Where:

- `lon, lat` are geographic coordinates of counties (normalized)
- `t` is time (normalized)
- `u` is the normalized case count (cases / population)

---

## 🚀 How to Run

### Step 1: Prepare the Environment

```bash
pip install -r requirements.txt  # Or install: pandas, geopandas, torch, scikit-learn, matplotlib
```

### Step 2: Generate the Dataset

```bash
python scripts/nyt_to_csv.py
```

This will create `data/covid_county_cases.csv` from raw NYT, geometry, and population files.

### Step 3: Run Exploratory Analysis

Open:

```bash
EDA.ipynb
```

To explore spatial and temporal trends in the dataset.

### Step 4: Train the PINN

```bash
python main_train.py
```

This will train the model and save `checkpoints/model.pth`.

### Step 5: Evaluate the Model

Open:

```bash
evaluate.ipynb
```

This notebook:

- Calls `main()` to train
- Loads the model
- Computes RMSE
- Plots predicted surfaces

---

## 📌 Features

- ✅ Modular design
- ✅ Real spatiotemporal COVID data
- ✅ Population normalization
- ✅ Collocation-based residual loss
- ✅ Evaluation and RMSE analysis
- ✅ Extensible for more PDEs or pandemic models

---

## 📄 Reference: Project Proposal

A full PDF proposal is available in `docs/Project_Proposal.pdf` (you can move the original shared PDF here). It includes:

- Motivation
- Mathematical formulation
- PINN architecture
- Training framework
- Future directions

---

## 🧪 Requirements

- Python 3.8+
- PyTorch
- pandas, matplotlib, geopandas

---

## 📬 Contact

If you have questions or want to contribute, please reach out!

---

## ✅ TODO (Optional Extensions)

-

