# 🌸 AI/ML Internship — Task 1: Iris Dataset EDA

## Task Objective
Explore and visualize the Iris dataset to understand data distributions, feature relationships, and potential outliers using `pandas`, `matplotlib`, and `seaborn`.

## Dataset Used
- **Name:** Iris Dataset
- **Source:** `seaborn.load_dataset('iris')` (originally from UCI ML Repository)
- **Size:** 150 samples × 5 columns
- **Target Variable:** `species` (setosa, versicolor, virginica)
- **Features:** sepal_length, sepal_width, petal_length, petal_width

## Models Applied
> No model training in this task — pure EDA and visualization.

## Visualizations Produced
| Plot | Purpose |
|---|---|
| Pairplot (scatter matrix) | All feature relationships by species |
| Focused scatter plots | Best vs. worst feature pairs |
| Histograms + KDE | Feature value distributions per species |
| Box plots + Strip plots | Outlier detection per species |
| Correlation heatmap | Linear relationships between features |
| Violin plots | Distribution shape and spread |

## Key Results & Findings
- **Petal features** (`petal_length`, `petal_width`) are the best discriminators — clean cluster separation
- **Iris setosa** is linearly separable from the other two species
- **Versicolor vs Virginica** overlap in sepal space — harder to separate
- Petal length and width are highly correlated (r = 0.96)
- No missing values; dataset is perfectly balanced (50 samples/class)
- Minor outliers in `sepal_width` for setosa

## Requirements
```bash
pip install pandas numpy matplotlib seaborn
```

## How to Run
```bash
jupyter notebook Task1_Iris_EDA.ipynb
```

## Repository Structure
```
📦 Developershub-Cooperation-ai-internship/
 ┣ 📓 task1_iris_visualization.ipynb   # Main notebook
 ┗ 📄 README.md              # This file
```