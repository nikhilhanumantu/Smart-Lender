# Smart Lender AI - ML-Powered Loan Eligibility & Analytics Platform

Smart Lender AI is an end-to-end, machine learning-driven web application designed to evaluate loan applications, predict approval eligibility, calculate dynamic risk scores, and present deep data analytics. Built on Python/Flask and styled with Tailwind CSS, this project provides a premium user experience coupled with state-of-the-art predictive performance.

---

## 🌟 Key Features

### 1. **Intelligent Predictive Model**
* Powered by machine learning classifiers (including **XGBoost, Random Forest, Decision Trees, and K-Nearest Neighbors**).
* Balanced dataset using **SMOTE** (Synthetic Minority Over-sampling Technique) to ensure unbiased predictions.
* Automatically scales features and loads model binaries on startup.

### 2. **Interactive Loan Eligibility Calculator**
* Dynamic **Eligibility Calculator** validating 11 key applicant metrics (Income, Co-applicant Income, Credit History, Dependents, etc.).
* Implements a **dynamic credit scorecard (0-100)** representing approval confidence.
* Real-time financial calculations, including **Estimated Monthly Installments (EMI)**, **Interest Rate calculation**, and the **EMI-to-Income ratio**.
* Auto-generated **actionable risk warnings** (e.g., low credit history warnings, high EMI-to-income warnings).

### 3. **Exploratory Data Analysis (EDA)**
* Deep-dive charts visualizing dataset distributions and patterns.
* Categorized into:
  * **Univariate analysis** (distributions of income, credit history, gender, property area, etc.)
  * **Bivariate analysis** (status correlations, income vs. loan size, dependents vs. status)
  * **Multivariate analysis** (pairplots, correlation heatmaps, heatmaps by education and marriage status)

### 4. **Executive Analytics Dashboard**
* Real-time dataset insights (total applications, overall approval rate, average applicant income, and loan amounts).
* Side-by-side **Model Performance Comparison** chart showing Accuracy, Precision, Recall, and F1-score across evaluated algorithms.

### 5. **Robust Security & Routing**
* Protected result routes preventing page skipping.
* Validation safeguards and input sanitization to prevent application crashes under bad data inputs.

---

## 📁 Repository Structure

The project has been organized into clear project phases and code subfolders:

```text
Smart Lender/
├── 1. Brainstorming & Ideation/     # Problem definitions, Empathy maps, Brainstorming templates
├── 2. Requirement Analysis/         # Customer Journey, Data Flow, Tech stack specs, Solution requirements
├── 3. Project Design Phase/         # Problem-Solution Fit, Proposed Solution, Solution Architecture
├── 4. Project Planning Phase/       # Project Planning and milestone roadmaps
├── 5. Project Development Phase/    # Project Development source directory
├── 6. Project Testing/              # Performance/QA testing templates and plans
├── 7. Project Documentation/        # Sample documentation and executable layout files
├── 8. Project Demonstration/        # Presentations, demo schedules, and scalability plans
└── Code Files/                      # Core application source folder
    ├── dataset/                     # Contains 'loan_approval_dataset.csv'
    ├── static/                      # CSS stylesheets, Tailwind configurations, and generated EDA charts
    ├── templates/                   # Flask HTML layout templates
    ├── tests/                       # Automated tests (Python unittest & Postman Collections)
    ├── app.py                       # Main Flask web application server
    ├── train_pipeline.py            # Model training, evaluations, SMOTE balancing, and chart export script
    ├── tailwind.config.js           # Tailwind CSS theme configurations
    └── TESTING_REPORT.md            # Execution summaries, test matrices, and QA logs
```

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have **Python 3.8+** installed.

### 2. Setup Virtual Environment
Run the following commands in your terminal:
```bash
# Clone the repository (if not already done)
git clone https://github.com/nikhilhanumantu/Smart-Lender.git
cd Smart-Lender/Code\ Files

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install flask pandas numpy scikit-learn xgboost imbalanced-learn matplotlib seaborn
```

### 4. Run the Machine Learning Pipeline
Train the classifiers, balance classes using SMOTE, output EDA images, and serialize the best performing model:
```bash
python train_pipeline.py
```
This saves `best_model.pkl` and creates visualizations under `static/images/charts/`.

### 5. Start the Web Application
```bash
python app.py
```
Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your web browser.

---

## 🧪 Testing Framework

The platform implements a multi-layered testing workflow to guarantee application safety and scalability. Detailed logs are available in [TESTING_REPORT.md](file:///c:/Users/Dell/OneDrive/Desktop/Smart%20Lender/Code%20Files/TESTING_REPORT.md).

### 1. Backend Integration Tests
Executes the Flask router assertions, validation checks, session controls, and predict handlers:
```bash
python -m unittest tests/test_app.py
```

### 2. API Integrity Tests (Postman)
* Postman Collection version: `2.1`
* File: `tests/smart_lender.postman_collection.json`
* Runs external POST requests against `/predict` to validate json payload compatibility.

### 3. Concurrency & Performance Load Tests (Locust)
* File: `locustfile.py`
* Tests routing latency, RPS, and concurrent user loads.

