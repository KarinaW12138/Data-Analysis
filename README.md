Stroke Risk Prediction Under Severe Class Imbalance



Project Overview
This project evaluates classification models on a highly imbalanced clinical dataset (n=5,110, minority class ≈ 5%). The objective is to benchmark an interpretable baseline (Logistic Regression) against an ensemble learning approach (Random Forest) in predicting stroke risk. 

Given the severe class imbalance, model performance is primarily evaluated using Precision-Recall Area Under the Curve (PR-AUC) rather than standard accuracy, alongside an analysis of optimal decision thresholds for clinical screening applications.

 Key Skills Demonstrated
Statistical Modeling:** Logistic Regression, Random Forest (`caret`, `randomForest`).
Imbalanced Data Handling:** Moving beyond accuracy; utilizing PR-AUC, F1-Score, and Recall (`PRROC`, `pROC`).
Clinical Threshold Optimization:** Adjusting decision boundaries (0.50 vs. 0.10) to minimize false negatives in medical screening.
Data Preprocessing:** Median imputation for continuous variables, robust factor encoding.

Key Findings
1. The Trap of Accuracy: At the standard 0.50 threshold, both models predicted "No Stroke" for virtually all patients, yielding an artificially high accuracy of ~95.18% but an unacceptable Recall of 1.3%.
2. Interpretable Models vs. Ensembles: The baseline Logistic Regression model achieved a PR-AUC of 0.1789 compared to 0.1642 for Random Forest, and a higher AUC-ROC (0.8399 vs. 0.8101). This demonstrates that a well-calibrated, interpretable model can match or outperform complex tree-based ensembles when the dataset is highly imbalanced.
3. Threshold Adjustment is Critical: Lowering the classification threshold to 0.10 allowed Logistic Regression to correctly identify 45 of 75 stroke cases (Recall = 60.0%), proving that threshold adjustment alone provides a massive, cost-free improvement in identifying high-risk patients.

Repository Structure
 `Predictive Stroke Analysis Data 2 (1).R`: The combined, fully commented R script containing all data manipulation, modeling, and evaluation code.
 `stroke_analysis.Rmd`: The RMarkdown file used to generate the final HTML report.
 `index.html`: The compiled interactive report (viewable via the link at the top).
 `visualizations`: Folder containing high-resolution `.png` exports of the PR-Curves, Variable Importance, and EDA plots.
