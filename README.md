# PINN-based COVID-19 Modeling using NYT Data

This project demonstrates how to use **Physics-Informed Neural Networks (PINNs)** to model the spatiotemporal spread of COVID-19 in the U.S. using real-world data. We preprocess the dataset, train a PINN model, compare it against traditional numerical solvers, and visualize the results.

---

## 🧠 Project Summary

This work is based on applying **PINNs** to a real epidemiological problem — modeling COVID-19 spread across U.S. counties using:

- The **2D spatial domain** (longitude and latitude)
- The **temporal evolution** (time axis)
- A **PDE model** for disease diffusion dynamics

PINNs are trained on real data and guided by the governing partial differential equations (e.g., the heat equation). This allows us to incorporate physical knowledge into machine learning, improving generalization and robustness.


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

The raw COVID-19 case data comes from:

📥 **New York Times COVID-19 dataset** (U.S. counties):  
🔗 https://github.com/nytimes/covid-19-data

Other supporting datasets:
- **U.S. Census Shapefiles**: https://www2.census.gov/geo/tiger/
- **County Population Estimates**: https://www.census.gov/data/datasets/time-series/demo/popest/2020s-counties-total.html

**Note**: The processed CSV file is too large to store in GitHub.  
Use the provided script below to regenerate it
---

## 🚀 How to Run the Project

### Step 1. Clone the Repo

```bash
git clone https://github.com/yourusername/pinn_project.git
cd pinn_project

### Step 2: Prepare the Environment

```bash
pip install -r requirements.txt  # Or install: pandas, geopandas, torch, scikit-learn, matplotlib
```

### Step 3: Generate the Dataset

```bash
python scripts/nyt_to_csv.py
```

This will create `data/covid_county_cases.csv` from raw NYT, geometry, and population files.

### Step 4: Run Exploratory Analysis

Open:

```bash
jupyter notebook EDA.ipynb
```

View heatmaps, temporal trends, and normalized COVID case surfaces.

### Step 5: Train the PINN

```bash
python main_train.py
```

- Trains the PINN on real data

- Applies residual loss from the PDE

- Saves to checkpoints/model.pth

### Step 6: Evaluate the Model

Open:

```bash
jupyter notebook evaluate.ipynb
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

## 🧪 Requirements Summary
Minimum Python 3.8+
- torch>=1.12
- numpy
- pandas
- matplotlib
- geopandas
- scikit-learn

Install with:

pip install -r requirements.txt

---

inference

🤝 Acknowledgments
- NYT Data Team

- U.S. Census Bureau

- Raissi et al. (2019), Physics-Informed Neural Networks (PINNs)

---

## 📬 Contact

If you have questions or want to contribute, please reach out!

---

## ✅ TODO (Optional Extensions)

-

