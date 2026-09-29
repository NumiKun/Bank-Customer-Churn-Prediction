# Bank Customer Churn Prediction and Customer Segmentation

An end-to-end unsupervised machine learning pipeline for banking customer segmentation, designed to identify distinct behavioral and financial cohorts, drive personalized retention strategies, and mitigate customer churn.

---

## Executive Summary

Customer churn poses a significant threat to profitability in retail banking. Retaining existing customers is substantially more cost-effective than customer acquisition. However, uniform retention strategies often fail because customers exhibit diverse financial behaviors, balances, product holdings, and demographic profiles.

This repository implements an enterprise-grade customer segmentation pipeline utilizing unsupervised learning (K-Means clustering validated by Hierarchical Agglomerative Clustering, PCA, and t-SNE). By partitioning a 10,000-customer dataset into 4 empirically validated segments, this system provides financial institutions with actionable intelligence for targeted cross-selling, customized engagement, and high-risk churn mitigation.

---

## Key Customer Segments and Business Profiles

The clustering pipeline discovered 4 distinct customer cohorts based on financial engagement, wealth level, product utilization, and demographics:

| Cluster | Segment Name | Customer Count | Population Share | Avg. Balance | Avg. Products | Zero Balance Pct. | Dominant Demographics | Strategic Focus |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **0** | Affluent Multi-Product Advocates | 2,022 | 20.2% | $120,418 | 2.13 | 0.0% | Avg. Age 36.8, Germany dominant (53.5%) | Loyalty rewards, premium tier services, wealth management cross-selling. |
| **1** | Affluent Single-Product Targets | 3,558 | 35.6% | $120,591 | 1.00 | 0.0% | Avg. Age 36.1, France (46.5%) & Germany (31.2%) | High-priority cross-sell targets; incentivize second product adoption to reduce churn risk. |
| **2** | Active Zero-Balance Everyday Users | 3,266 | 32.7% | $85 | 1.82 | 99.8% | Avg. Age 35.9, France (67.1%) & Spain (32.9%) | Deposit accumulation campaigns, low-cost savings boosters, transaction fee optimization. |
| **3** | Mature Wealth Preservation Clients | 1,154 | 11.5% | $79,751 | 1.32 | 31.1% | Avg. Age 59.9, distributed across France, Germany, Spain | Retirement planning, wealth preservation, specialized senior advisory, dedicated relationship managers. |

### Strategic Business Recommendations

1. **Mitigate Single-Product Attrition (Cluster 1):** Customers with only 1 banking product and high deposits present high churn vulnerability if competitor rates shift. Providing bundled credit card or insurance incentives directly strengthens account stickiness.
2. **Deposit Inflow Campaign for Low-Balance Users (Cluster 2):** Nearly one-third of the customer base maintains negligible balances while actively utilizing multiple products. Targeted payroll deposit incentives and high-yield micro-savings accounts can capture wallet share.
3. **High-Net-Worth Retention and Advisory (Cluster 0 and Cluster 3):** High-balance segments represent the core asset base. Priority customer support, tailored investment vehicles, and wealth advisory programs should be allocated to these groups.

---

## Machine Learning Pipeline Architecture

The pipeline follows a structured, modular workflow designed for reproducibility and integration:

1. **Data Ingestion and Integrity Verification:**
   - Ingestion of 10,000 retail banking records.
   - Rigorous schema validation, missing value verification, duplicate checks, and outlier inspection.
2. **Exploratory Data Analysis (EDA):**
   - Univariate, bivariate, and multivariate distribution analysis.
   - Bimodal balance distribution discovery (revealing a distinct zero-balance mass of 36% overall).
   - Correlation analysis across credit metrics, balance, tenure, and product counts.
3. **Feature Engineering and Standardization:**
   - Binary encoding for zero-balance behavior (`HasZeroBalance`).
   - One-hot encoding for geographic regions (`Geo_France`, `Geo_Germany`, `Geo_Spain`).
   - Binary transformation for gender attributes.
   - Z-score normalization using `StandardScaler` to ensure scale invariance across continuous and encoded features.
4. **Optimal Cluster Selection:**
   - Multi-metric evaluation across K = 2 to K = 10:
     - Within-Cluster Sum of Squares (WCSS / Elbow Method)
     - Silhouette Coefficient
     - Davies-Bouldin Index
     - Calinski-Harabasz Index
   - Selection of K = 4 as the mathematically and operationally optimal balance between cohesion and separation.
5. **Model Training and Cross-Validation:**
   - Fitting K-Means with k-means++ initialization (`n_init=20`).
   - Validation against Hierarchical Agglomerative Clustering (Ward linkage) via dendrogram analysis.
6. **Dimensionality Reduction and Spatial Inspection:**
   - Principal Component Analysis (PCA) 2D projection and eigenvector loading analysis.
   - t-Distributed Stochastic Neighbor Embedding (t-SNE) for non-linear manifold separation.
7. **Artifact Persistence:**
   - Serialization of models, preprocessors, label arrays, evaluation summaries, and enriched customer datasets.

---

## Model Evaluation Metrics

| Metric | Value | Interpretation |
| :--- | :---: | :--- |
| **Optimal K** | 4 | Selected via elbow curvature and cluster balance |
| **Total Customers Evaluated** | 10,000 | Full dataset coverage without data loss |
| **Input Feature Dimensionality** | 10 | Standardized numerical and encoded categorical features |
| **Within-Cluster Sum of Squares (WCSS)** | 39,456.74 | Compact intra-cluster dispersion |
| **Silhouette Score** | 0.1918 | Realistic separation for high-dimensional mixed retail banking data |
| **Davies-Bouldin Index** | 1.7911 | Lower ratio indicating sound cluster partition separation |
| **Calinski-Harabasz Score** | 1,821.86 | High variance ratio confirming distinct group means |

---

## Project Structure

```text
Bank-Customer-Churn-Prediction/
│
├── Dataset/
│   └── Customer-Churn-Records.csv        # Raw banking customer dataset (10,000 records)
│
├── Model/
│   ├── customer_segmentation.ipynb       # Complete end-to-end segmentation notebook
│   └── artifacts/                        # Persisted models, data, and visual reports
│       ├── kmeans_model.pkl              # Serialized trained K-Means model
│       ├── scaler.pkl                    # Serialized StandardScaler
│       ├── pca_model.pkl                 # Serialized PCA transformer
│       ├── cluster_labels.npy            # NumPy cluster assignments array
│       ├── clustered_customers.csv       # Customer dataset with cluster labels
│       ├── evaluation_results.json       # Quantitative performance metrics & profiles
│       ├── feature_names.json            # Ordered feature schema used in modeling
│       ├── cluster_selection_metrics.png # Elbow & Silhouette selection charts
│       ├── hierarchical_dendrogram.png   # Ward linkage hierarchical dendrogram
│       ├── viz_pca_clusters.png          # 2D PCA cluster projection
│       ├── viz_tsne_clusters.png         # 2D t-SNE cluster projection
│       ├── cluster_profile_analysis.png  # Segment profile comparison plots
│       └── ...                           # Additional EDA and feature loading plots
│
├── requirements.txt                      # Project dependency specification
├── LICENSE                               # MIT License
└── README.md                             # Project documentation
```

---

## Getting Started

### Prerequisites

- Python 3.10, 3.11, 3.12, 3.13, or 3.14
- pip package manager
- Git

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/NumiKun/Bank-Customer-Churn-Prediction.git
   cd Bank-Customer-Churn-Prediction
   ```

2. **Create and activate a virtual environment:**
   - **Linux / macOS:**
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
   - **Windows (PowerShell):**
     ```powershell
     python -m venv venv
     .\venv\Scripts\Activate.ps1
     ```

3. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Execution

Launch Jupyter Notebook or JupyterLab to execute the pipeline:

```bash
jupyter notebook Model/customer_segmentation.ipynb
```

All figures, processed datasets, and serialized model files will automatically update in `Model/artifacts/`.

---

## Serialized Artifacts Reference

The pipeline persists the following production-ready artifacts in `Model/artifacts/`:

| Artifact | Type | Description |
| :--- | :--- | :--- |
| `kmeans_model.pkl` | Binary (Joblib) | Trained K-Means clustering estimator. |
| `scaler.pkl` | Binary (Joblib) | Fitted StandardScaler with training mean and scale parameters. |
| `pca_model.pkl` | Binary (Joblib) | Fitted 2D PCA transformer for dimensional projection. |
| `cluster_labels.npy` | Binary (NumPy) | 1D array of length 10,000 containing assigned cluster IDs (0 to 3). |
| `clustered_customers.csv` | CSV | Enriched dataset containing all raw columns alongside assigned `Cluster`. |
| `evaluation_results.json` | JSON | Machine-readable cluster counts, numerical averages, and validation metrics. |
| `feature_names.json` | JSON | Exact feature order required when serving new inference data to the model. |

---

## Technology Stack

- **Data Manipulation:** `pandas`, `numpy`
- **Machine Learning & Preprocessing:** `scikit-learn` (StandardScaler, KMeans, AgglomerativeClustering, PCA, TSNE)
- **Hierarchical Statistics:** `scipy` (linkage, dendrogram)
- **Model Persistence:** `joblib`
- **Data Visualization:** `matplotlib`, `seaborn`
- **Execution Environment:** `Jupyter Notebook`, `ipykernel`

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for complete details.

---

## Author

Developed by **NumiKun**.  
Contributions, feedback, and issue submissions are welcome.
