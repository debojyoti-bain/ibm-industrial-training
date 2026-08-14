---

# PROJECT DOCUMENTATION REPORT

## **PROJECT TITLE:** Smartphone Price Category Prediction System Based on Hardware Features
**Training Program:** IBM Industrial Training (Data Science & Machine Learning)  
**Domain:** Artificial Intelligence / Consumer Technology Analytics  
**Algorithm Used:** Decision Tree Classifier (`criterion="entropy"`, `max_depth=4`)  
**Programming Language:** Python 3  
**Key Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn  

---

## 1. EXECUTIVE SUMMARY & PROJECT OVERVIEW

### 1.1 Objective
The primary objective of this project is to develop an intelligent Machine Learning classification model that analyzes a smartphone's hardware specifications (such as RAM, Storage/ROM, Battery Capacity, Processor, Display Size, and Camera Resolution) and predicts its market price segment category.

### 1.2 Problem Statement
In the modern smartphone market, pricing strategy is heavily dictated by hardware specifications. Consumers and retailers often struggle to assess whether a smartphone is priced fairly based on its physical components. This system automates the valuation process by classifying devices into four distinct market segments:
1. **Category 0 (Budget Phone):** Low-cost entry-level devices ($\text{₹}5,000 - \text{₹}12,000$)
2. **Category 1 (Mid-Range Phone):** Balanced performance devices ($\text{₹}12,000 - \text{₹}25,000$)
3. **Category 2 (Premium Mid-Range):** Upper mid-tier devices ($\text{₹}25,000 - \text{₹}45,000$)
4. **Category 3 (Flagship Phone):** High-end flagship devices ($\text{₹}45,000+$)

---

## 2. DATASET ARCHITECTURE & SPECIFICATIONS

The system utilizes a dataset (`data.csv`) containing technical specifications of smartphones available in the retail market.

### 2.1 Hardware Feature Description

| Feature Name | Data Type | Description & Unit |
| :--- | :--- | :--- |
| `No_of_sim` | Numerical | Number of SIM card slots (e.g., 1 or 2) |
| `Ram` | Numerical | System Memory size in Gigabytes (GB) |
| `Battery` | Numerical | Battery capacity in Milliampere-hours (mAh) |
| `Display` | Numerical | Screen diagonal size in Inches |
| `Camera` | Numerical | Primary rear camera resolution in Megapixels (MP) |
| `External_Memory` | Numerical | Maximum expandable SD card limit in GB |
| `Android_version` | Numerical | Operating System major version (e.g., 11, 12, 13) |
| `Inbuilt_memory` | Numerical | Internal Storage / ROM in GB (e.g., 64, 128, 256) |
| `fast_charging` | Numerical | Fast Charging power rating in Watts (W) |
| `Screen_resolution`| Numerical | Display resolution height in Pixels (720, 1080, 1440) |
| `company` | Categorical | Brand Manufacturer (e.g., Samsung, Apple, Xiaomi, Vivo) |
| `Processor` | Categorical | CPU architecture type (e.g., Octa Core, Deca Core) |

### 2.2 Target Variable
* **`Price_Category` (Class Labels: `0, 1, 2, 3`):** Generated using quantile discretization (`pd.qcut`) on the continuous `Price` attribute to ensure equal class balance across the dataset.

---

## 3. METHODOLOGY & PIPELINE ARCHITECTURE

The project follows a 16-step end-to-end Machine Learning pipeline:

```
[Raw CSV Dataset] ──> [Data Cleaning & Regex Extraction] ──> [Label Encoding] 
        │
        ├──> [Quantile Price Categorization] ──> [Train-Test Split (80/20)]
        │
        └──> [Decision Tree Model Training] ──> [Model Evaluation & Heatmaps]
        │
        └──> [Visual Flowchart Diagram] ──> [Interactive Console Predictor]
```

### 3.1 Key Pipeline Steps
1. **Library Importation:** Loading foundational scientific packages (`pandas`, `numpy`, `sklearn`, `matplotlib`, `seaborn`).
2. **Data Loading:** Reading `data.csv` into a Pandas DataFrame structure.
3. **Data Inspection:** Checking dataset dimensions (`df.shape`), row samples (`head()`, `tail()`), and data types (`dtypes`).
4. **Missing Value Audit:** Detecting null entries across all attributes using `df.isnull().sum()`.
5. **Data Cleaning & Regex Extraction:** Using Regular Expressions (`re`) to extract pure numerical values from text-filled columns (e.g., extracting `5000` from `"5000 mAh Battery"`).
6. **Categorical Encoding:** Converting text columns (`company`, `Processor`) into numeric labels using Scikit-Learn’s `LabelEncoder` after standardizing all strings to lowercase.
7. **Price Discretization:** Converting continuous Rupee values into 4 balanced price categories using `pd.qcut(df['Price'], q=4)`.
8. **Feature Selection:** Dropping non-hardware and post-purchase columns (`Name`, `Price`, `Rating`, `Spec_score`, `Processor_name`, `Unnamed: 0`) to prevent target leakage.
9. **Train-Test Division:** Splitting the dataset into 80% Training Data and 20% Testing Data using stratified sampling (`train_test_split`).
10. **Model Training:** Fitting a `DecisionTreeClassifier` configured with Entropy criterion and a depth limit of `max_depth=4`.
11. **Performance Testing:** Evaluating test sample predictions against true labels.
12. **Detailed Metrics:** Computing Confusion Matrix ($4 \times 4$) and Classification Report (Precision, Recall, F1-Score).
13. **Data Visualizations:** Generating a $2 \times 2$ grid containing Count Plot, Scatter Plot, Heatmap, and Feature Importance Bar Chart.
14. **Tree Flowchart Rendering:** Plotting the visual Decision Tree structure using `plot_tree`.
15. **Real-Time Prediction Engine:** Providing a user-friendly console prompt with auto-matching case-insensitive inputs.
16. **Project Conclusion:** Summarizing system execution results.

---

## 4. MACHINE LEARNING ALGORITHM: DECISION TREE CLASSIFIER

### 4.1 Theoretical Concept
A **Decision Tree** is a non-parametric supervised learning algorithm that breaks down a complex decision-making process into a series of simpler binary or multi-way split rules based on feature attributes.

### 4.2 Mathematical Splitting Criterion (Entropy & Information Gain)
The model selects feature split thresholds by maximizing **Information Gain** ($IG$), which measures the reduction in uncertainty (Entropy):

$$\text{Entropy } H(S) = - \sum_{i=1}^{k} p_i \log_2 (p_i)$$

$$\text{Information Gain } IG(S, A) = H(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} H(S_v)$$

*Where:*
* $H(S)$ is the impurity of the current node dataset $S$.
* $p_i$ is the probability of a sample belonging to price category $i$.
* $IG(S, A)$ represents the entropy reduction achieved by splitting dataset $S$ on hardware feature $A$.

### 4.3 Hyperparameter Configuration
* **`criterion="entropy"`:** Selected to split nodes based on Information Gain calculations.
* **`max_depth=4`:** Restricted to 4 levels to prevent tree overfitting and maintain a readable visual diagram.
* **`random_state=42`:** Fixes the pseudo-random number generator for consistent model training.

---

## 5. EXPERIMENTAL RESULTS & EVALUATION

### 5.1 Model Accuracy
* **Accuracy Score:** `~0.6825` (68.25%)
* **Contextual Evaluation:** In a 4-class classification problem where random guessing yields a $25\%$ success baseline ($100\% / 4$), an accuracy of $68.25\%$ demonstrates strong predictive performance, operating nearly **three times better than random chance**.

### 5.2 Confusion Matrix Analysis ($4 \times 4$)
The $4 \times 4$ confusion matrix evaluates true vs. predicted classifications across the four categories:

$$\begin{bmatrix}
C_{0,0} & C_{0,1} & C_{0,2} & C_{0,3} \\
C_{1,0} & C_{1,1} & C_{1,2} & C_{1,3} \\
C_{2,0} & C_{2,1} & C_{2,2} & C_{2,3} \\
C_{3,0} & C_{3,1} & C_{3,2} & C_{3,3}
\end{bmatrix}$$

* **Main Diagonal ($C_{i,i}$):** Represents correctly classified smartphones for each category.
* **Off-Diagonal Elements:** Represent misclassifications (e.g., a Mid-Range phone predicted as Premium).

---

## 6. SYSTEM VISUALIZATIONS

The project includes five distinct graphical outputs generated via Matplotlib and Seaborn:

1. **Price Category Distribution (Count Plot):** Confirms balanced class distribution across Budget, Mid-Range, Premium, and Flagship categories.
2. **RAM vs. Storage Scatter Plot:** Displays hardware clustering, demonstrating that high-RAM ($8\text{GB}+$) and high-ROM ($256\text{GB}+$) devices concentrate heavily in Premium and Flagship categories.
3. **Hardware Feature Correlation Heatmap:** Displays pairwise linear correlations across all continuous specifications.
4. **Feature Importance Bar Plot:** Ranks the hardware attributes that contributed most to the Decision Tree's split decisions (e.g., RAM, Storage, and Screen Resolution).
5. **Decision Tree Flowchart Diagram:** A tree visualization displaying decision nodes, threshold values, entropy scores, sample counts, and class colors.

---

## 7. INTERACTIVE USER PREDICTION ENGINE

The system features an interactive console module (Step 15) allowing users to enter custom smartphone specifications in real-time.

### 7.1 Sample Input Execution
* **RAM:** `12 GB`
* **Internal Storage:** `512 GB`
* **Battery:** `5000 mAh`
* **Camera:** `108 MP`
* **Fast Charging:** `120 W`
* **Processor:** `Octa Core`
* **Brand:** `Samsung`

### 7.2 Output Generated
* **Predicted Category Code:** `3`
* **Predicted Price Segment:** `Flagship Phone`
* **Estimated Rupee Price Range:** `₹45,000+ (High Cost)`

---

## 8. CONCLUSION & FUTURE SCOPE

### 8.1 Conclusion
The **Smartphone Price Category Prediction System** successfully demonstrates the application of Decision Tree classification to real-world consumer technology datasets. By pre-processing text specifications into numeric attributes, the model accurately categorizes smartphones into four market tiers based on hardware specs.

### 8.2 Future Enhancements
1. **Web Interface Integration:** Deploying the trained model using Python Flask/Streamlit to create a web application.
2. **Price Regression:** Expanding the classification model into a continuous regression model to predict exact Rupee values.
3. **Ensemble Models:** Integrating Random Forest and XGBoost algorithms to further improve accuracy.

---
**Report Compiled for:** IBM Industrial Training Evaluation  
**Status:** Completed & Validated
