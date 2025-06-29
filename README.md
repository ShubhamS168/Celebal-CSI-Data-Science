
# 🔍 CSI Weekly Assignments Repository

Welcome to the **CSI (Celebal Summer Internship)** Weekly Assignments repository!  
This repository contains weekly tasks and programming exercises aimed at developing strong problem-solving and coding skills.

---

## 📁 Repository Structure

Assignments are organized by week. Each folder contains the Python scripts and a `README.md` documenting the task.


```

Celebal-CSI-Data-Science/Assignment/
 ┣ Week 1/
 ┃ ┣ upper_triangle.py
 ┃ ┣ lower_triangle.py
 ┃ ┣ pyramid.py
 ┃ ┗ README.md
 ┃ 
 ┣ Week 2/
 ┃ ┣ linked_list.py
 ┃ ┗ README.md
 ┃ 
 ┣ Week 3/
 ┃ ┣ cleaning_dataset.ipynb
 ┃ ┣ ipl_2024.csv
 ┃ ┣ IPL_cleaned_2024.csv
 ┃ ┣ ipl2024_visualizations.py
 ┃ ┣ visualizations/
 ┃ ┗ README.md
 ┃ 
 ┣ Week 4/
 ┃ ┣ IPL_2024_EDA.ipynb 
 ┃ ┣ IPL_cleaned_2024.csv
 ┃ ┣ ipl_2024_eda.py
 ┃ ┣ visualizations/
 ┃ ┗ README.md
 ┃ 
 ┣ Week 5/
 ┃ ┣ house_price_predictions.ipynb
 ┃ ┣ house_price_predictions.csv
 ┃ ┣ requirements.txt
 ┃ ┣ data/
 ┃ ┣ visualizations/
 ┃ ┗ README.md
 ┃ 
 ┣ Week 6/
 ┃ ┣ data/
 ┃ ┣ notebooks/
 ┃ ┣ outputs/
 ┃ ┣ src/
 ┃ ┣ visualizations/
 ┃ ┣ main.py
 ┃ ┣ ml_pipeline.log
 ┃ ┣ README.md
 ┃ ┗ requirements.txt
 ┃ 
 ┗ README.md

```

---

## 📌 Weekly Assignments

### 🗓️ Week 1: Triangle Pattern Generator

**Objective:**  
Implement three different triangle pattern generators that take dynamic user input.

**Included Patterns:**

- 🔼 Upper Triangle
- 🔽 Lower Triangle
- 🔺 Pyramid

**Features:**

- Accepts user-defined height/size
- Clear console outputs for patterns
- Input validation and error handling
- Logs pattern output for verification or debugging

**To Run:**

```bash
cd Celebal-CSI-Data-Science/Assignment/week\ 1
python upper_triangle.py
```

---

### 🗓️ Week 2: Singly Linked List (Object-Oriented)

**Objective:**  
Create a basic implementation of a singly linked list using object-oriented programming in Python.

**Features:**

- Class-based design with `Node` and `LinkedList` classes
- Core methods:
  - `add_node(data)`
  - `print_list()`
  - `del_nth_node(n)`
- Uses Python `typing.Optional` for type hinting
- Strong input validation and exception handling
- Visual explanation with comments and ASCII diagrams

**To Run:**

```bash
cd Celebal-CSI-Data-Science/Assignment/week\ 2
python linked_list.py
```

---

### 🗓️ Week 3: IPL 2024 Data Cleaning & Visualization

**Objective:**  
Analyze and visualize IPL 2024 player-wise match data using Python’s data science ecosystem.

**Features:**

- Raw dataset (`ipl_2024.csv`) cleaned using `cleaning_dataset.ipynb`
- Cleaned data saved as `IPL_cleaned_2024.csv`
- Automated visualizations generated via `ipl2024_visualizations.py`
- 13 insightful plots saved inside the `visualizations/` folder

**Topics Covered:**

- Data preprocessing and feature extraction
- GroupBy analytics for players and teams
- Performance metrics like strike rate, runs, wickets, boundaries, etc.
- Visualization using Seaborn and Matplotlib

**To Run:**

```bash
cd Celebal-CSI-Data-Science/Assignment/week\ 3
python ipl2024_visualizations.py
```

---

### 🗓️ Week 4: IPL 2024 – Exploratory Data Analysis (EDA) 

**Objective:**  
Perform in-depth EDA on IPL 2024 dataset to uncover patterns, relationships, and hidden insights in player and team performances.

**Features:**

- Cleaned dataset (`IPL_cleaned_2024.csv`) used for visual analysis
- Exploratory insights using histograms, boxplots, scatter plots, correlation maps, and pairplots
- 10 advanced visualizations saved inside the `visualizations/` folder

**Topics Covered:**

- Distribution of numeric features
- Relationship between runs and balls faced
- Team-level average strike rate
- Correlation between batting features
- Multivariate pairwise relationships among player metrics

**To Run:**

```bash
cd Celebal-CSI-Data-Science/Assignment/week\ 4
python ipl_2024_eda.py
```
---

### 🗓️ Week 5: House Price Prediction – Data Preprocessing & Feature Engineering

**Objective:**  
Build a regression model that predicts housing prices using the Ames Housing dataset. This includes complete data preprocessing, feature engineering, and applying multiple regression techniques.

**Features:**

- Uses Kaggle’s Ames Housing dataset (`train.csv`, `test.csv`)
- Built models: Linear, Ridge, Lasso, Random Forest, XGBoost, SVR, KNN
- Engineered features like `TotalSF`, `AgeOfHouse`, `QualityIndex`
- Output predictions saved as `house_price_predictions.csv`
- Evaluated with RMSE, MAE, and R² metrics

**Topics Covered:**

- EDA with histograms, boxplots, scatter plots  
- Correlation heatmap to identify key numeric relationships  
- Sale Price trends by Living Area & Quality
- Missing value handling and encoding techniques  
- Feature importance ranked via Random Forest


**To Run:**

```bash
cd Celebal-CSI-Data-Science/Assignment/week\\ 5
jupyter notebook house_price_predictions.ipynb
```

---

### 📊 Week 6: Model Evaluation and Hyperparameter Tuning – Student Performance Classification

**Objective:**  
Train multiple machine learning models and evaluate their performance using metrics such as accuracy, precision, recall, and F1-score. Implement hyperparameter tuning techniques like GridSearchCV and RandomizedSearchCV to optimize model parameters.

**Features:**

- UCI Student Performance dataset (649 samples, 30+ features)
- Classification into 4 performance categories based on final grade (G3)
- Built models: Logistic Regression, Decision Tree, Random Forest, SVM, KNN, XGBoost, Naive Bayes
- Evaluation metrics: Accuracy, Precision, Recall, F1-Score
- Visualizations: Correlation heatmap, G3 distribution, performance bar chart, confusion matrices
- Hyperparameter tuning with RandomizedSearchCV
- Outputs stored in /outputs: .csv files, .pkl models, visual charts

**Topics Covered:**

- Label engineering and train-test split
- Model training with 5-fold cross-validation
- Model comparison on held-out test set
- Hyperparameter tuning for top models
- Visual and tabular analysis of model performance 


**To Run:**

```bash
cd Celebal-CSI-Data-Science/Assignment/Week\ 6
python main.py
```

---

## ⚙️ Highlights

- 🧠 Logical and structured problem-solving
- 🧪 Input validation and defensive programming
- 🔁 Weekly assignment-based workflow
- 📝 Clean, readable code with documentation

---

## 🚀 Getting Started

1. **Clone the repository**

```bash
git clone https://github.com/ShubhamS168/Celebal-CSI-Data-Science.git
```

2. **Navigate to a specific week**

```bash
cd Celebal-CSI-Data-Science/Assignment/week\ 1
```

3. **Run any Python file**

```bash
python pyramid.py
```

---

## ✍️ Author

**Shubham Sourav**  
*Data Science Intern at Celebal Technologies*

---

## 📬 Contact

For any queries, feedback, or collaboration, feel free to connect:

📧 **Email:** [shubhamsourav475@gmail.com](mailto:shubhamsourav475@gmail.com)

---

> 📝 **Note:**  
> This repository is maintained as part of the CSI (Celebal Summer Internship) program and is intended for educational use.
