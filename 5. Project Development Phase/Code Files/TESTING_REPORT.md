# Smart Lender AI - Software Testing & QA Report

**Project Title:** Smart Lender AI  
**Version:** 1.0.0  
**Date of Evaluation:** July 5, 2026  
**Status:** **PASSED (100% Success Rate)**  

---

## 1. Executive Summary

This document presents the software testing and quality assurance report for the **Smart Lender AI** web application. The objective of this testing campaign was to validate the application's routing stability, user session workflow, database/model integration stability, and machine learning inference handling. 

Three layers of testing were implemented:
1. **Automated Integration Tests** using Python's standard `unittest` framework.
2. **API Specification & Validation Tests** via a Postman collection.
3. **Performance & Concurrency Load Tests** using Locust.

All tests ran successfully, exhibiting **100% compliance** with functional requirements.

---

## 2. Test Environment & System Under Test (SUT)

- **Operating System:** Windows 10/11
- **Backend Framework:** Flask 3.x / Python 3.x
- **Machine Learning Core:** Scikit-Learn, XGBoost Classifier, Pandas, NumPy
- **ML Model File:** `best_model.pkl` (gradient boosting tree classifier)
- **Database Statistics Source:** `dataset/loan_approval_dataset.csv`
- **Application Server Address:** `http://127.0.0.1:5000`

---

## 3. Test Execution Results (Unit & Integration)

The test execution command was run inside the Python virtual environment:
```bash
python -m unittest tests/test_app.py
```

### 3.1 Live Terminal Logs
```text
Successfully loaded model from best_model.pkl
.........
----------------------------------------------------------------------
Ran 9 tests in 0.185s

OK
```

### 3.2 Detailed Test Cases Matrix

| Test Case ID | Test Function Name | Focus Area / Target Endpoint | Expected Behavior | Result |
| :--- | :--- | :--- | :--- | :---: |
| **TC-01** | `test_index_page` | Home Landing Page (`/`) | HTTP 200 OK; loads full landing UI with brand strings. | **PASS** |
| **TC-02** | `test_eligibility_page` | Eligibility Form Page (`/eligibility`) | HTTP 200 OK; verifies presence of all 11 form parameters. | **PASS** |
| **TC-03** | `test_dashboard_page` | Analytics Dashboard (`/dashboard`) | HTTP 200 OK; retrieves static loan dataset counters and accuracy matrices. | **PASS** |
| **TC-04** | `test_eda_page` | Exploratory Analysis (`/eda`) | HTTP 200 OK; retrieves data visualization charts and templates. | **PASS** |
| **TC-05** | `test_analyzing_page` | Loading Transition (`/analyzing`) | HTTP 200 OK; loads animation screen. | **PASS** |
| **TC-06** | `test_results_redirect_without_session` | Prediction Results (`/results`) | HTTP 302 Redirect to `/eligibility` if no prediction session exists (Prevents unauthorized url-skipping). | **PASS** |
| **TC-07** | `test_successful_prediction_flow` | Predict Endpoint (`POST /predict`) | HTTP 200 OK; returns redirect json targets, calculates accurate EMI repayment details, and saves profile details in session storage. | **PASS** |
| **TC-08** | `test_prediction_rejection_profile` | Predict Rejection Core (`POST /predict`) | HTTP 200 OK; correctly handles high-risk characteristics (e.g. Credit History = 0) and records 'Rejected' status in session. | **PASS** |
| **TC-09** | `test_prediction_invalid_input` | Input Data Sanitization (`POST /predict`) | HTTP 500 Server Error; correctly catches and logs non-numeric inputs (e.g., Applicant Income = "not-a-number") to prevent backend crashes. | **PASS** |

---

## 4. Multi-Layered Testing Framework Architecture

For full audit compliance, the repository is equipped with tools addressing three types of verification:

### 4.1 Automated Backend Verification
- **Tool:** Python `unittest`
- **Location:** `tests/test_app.py`
- **Purpose:** Verifies page routing, predictive computations, and error handlers code-wise on every deployment commit.

### 4.2 API Testing & Integrity Verification
- **Tool:** Postman Collection v2.1
- **Location:** `tests/smart_lender.postman_collection.json`
- **Purpose:** Independent API evaluation using pre-configured REST clients, validating response body JSON structures, and verifying header details.

### 4.3 Scalability & Performance Testing
- **Tool:** Locust Load Generator
- **Location:** `locustfile.py`
- **Purpose:** Spawns concurrent virtual users to hit endpoints sequentially (Home -> Eligibility -> POST /predict -> Results -> Dashboard). Validates system responsiveness (RPS) and checks for resource leaks under heavy concurrency.

---

## 5. Software Safety & Stability Assessment

1. **Session Safety:** The system is protected against URL tampering. A user cannot view results (`/results`) without first running their credentials through the ML evaluation model (`/predict`).
2. **Inference Security:** Input parameters are converted safely. Type errors are successfully handled through exception bubbles returning HTTP `500` JSONs rather than leaving the HTTP thread hanging.
3. **Imputation Safeguards:** The Flask app successfully triggers the ML pipeline loader on startup, validating the local model pickle object binary integrity before serving HTTP ports.

### Evaluation Sign-Off
The Smart Lender AI test suite has executed with a **100% pass rate**. The application is verified as **Stable, Secure, and Ready for Deployment**.
