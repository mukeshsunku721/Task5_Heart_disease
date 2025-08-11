# Task 5: Decision Trees & Random Forests — Heart Disease Prediction

## 🎯 Objective
The goal of this task is to learn **tree-based models** for classification using the **Heart Disease Dataset** and compare the performance of **Decision Trees** vs **Random Forests**. We also explore **overfitting**, **tree depth control**, and **feature importance**.

---

## 📦 Tools Used
- **Scikit-learn** — Model building & evaluation  
- **Pandas** — Data loading & preprocessing  
- **Graphviz** — Visualizing Decision Trees  
- **Matplotlib** — Plotting results  

Dataset: [Heart Disease Dataset (Kaggle)](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)

---

## 📋 Steps Performed
1. **Load Dataset** from CSV  
2. **Train/Test Split** (80/20 split)  
3. **Train a Decision Tree Classifier** with default parameters  
4. **Visualize the tree** using Graphviz  
5. **Limit max depth** to prevent overfitting  
6. **Train a Random Forest Classifier** and compare results  
7. **Cross-validation** for stability check  
8. **Evaluate with precision, recall, and F1-score**  
9. **Interpret feature importances**  

---

## 📊 Model Results

| Model                              | Accuracy  | Precision (Class 0 / Class 1) | Recall (Class 0 / Class 1) | F1-score (Weighted) | Cross-validation Accuracy |
|------------------------------------|-----------|--------------------------------|----------------------------|---------------------|---------------------------|
| Decision Tree (default)            | **98.54%** | 0.97 / 1.00                    | 1.00 / 0.97                | 0.99                 | 1.0000                    |
| Decision Tree (Max Depth = 3)      | 78.05%    | -                              | -                          | -                   | -                         |
| Random Forest                      | **98.54%** | 0.97 / 1.00                    | 1.00 / 0.97                | 0.99                 | 0.9971                    |

---

---

## 🔍 Insights
- Shallower trees (max depth = 3) reduce overfitting but lose accuracy.  
- Random Forest matches the performance of the best Decision Tree but is more stable and less prone to overfitting due to averaging multiple trees.  
- Cross-validation scores confirm Random Forest’s better generalization compared to a single deep Decision Tree.  

---

## 📈 Feature Importance
Random Forest feature importance revealed the top predictors for heart disease:

1. **cp** — Chest pain type  
2. **thal** — Thalassemia  
3. **ca** — Number of major vessels  
4. **oldpeak** — ST depression induced by exercise  
5. **thalach** — Maximum heart rate achieved  

---

## 📌 Final Output
The final model is a **Random Forest Classifier** with **98.54% accuracy**, capable of predicting whether a patient has heart disease based on medical attributes.

---

## 🌳 Decision Tree Visualization (Graphviz)

A decision tree can be exported and visualized to show how splits are made at each node.  
Example code to visualize:  

```python
from sklearn.tree import export_graphviz
import graphviz

# Export tree to DOT format
dot_data = export_graphviz(
    dt_model,  
    out_file=None,  
    feature_names=X.columns,  
    class_names=['No Heart Disease', 'Heart Disease'],  
    filled=True,  
    rounded=True,  
    special_characters=True
)

# Render tree
graph = graphviz.Source(dot_data)
graph.render("decision_tree_visualization", format="png", cleanup=True)
graph

