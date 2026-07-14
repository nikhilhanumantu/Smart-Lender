import unittest
import os
import json
from app import app, load_model, model_data

class SmartLenderTestCase(unittest.TestCase):
    def setUp(self):
        """Set up testing client and configure Flask app for testing."""
        app.config['TESTING'] = True
        app.config['SECRET_KEY'] = 'test-secret-key'
        self.client = app.test_client()
        # Ensure model is loaded if available
        load_model()

    def test_index_page(self):
        """Verify that the home page loads successfully."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Smart Lender', response.data)

    def test_eligibility_page(self):
        """Verify that the eligibility form page loads successfully."""
        response = self.client.get('/eligibility')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'eligibility-form', response.data)

    def test_dashboard_page(self):
        """Verify that the dashboard page loads successfully and displays stats."""
        response = self.client.get('/dashboard')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Approved Loans', response.data)
        self.assertIn(b'Model Performance Comparison', response.data)

    def test_eda_page(self):
        """Verify that the exploratory data analysis page loads successfully."""
        response = self.client.get('/eda')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Exploratory Data Analysis', response.data)

    def test_analyzing_page(self):
        """Verify that the analyzing loading screen loads successfully."""
        response = self.client.get('/analyzing')
        self.assertEqual(response.status_code, 200)

    def test_results_redirect_without_session(self):
        """Verify that accessing results without performing a prediction redirects to the form."""
        response = self.client.get('/results')
        self.assertEqual(response.status_code, 302)
        self.assertTrue(response.location.endswith('/eligibility'))

    def test_successful_prediction_flow(self):
        """Test POST /predict with valid form data and verify it redirects and saves values in session."""
        if model_data is None:
            self.skipTest("Skipping prediction test because best_model.pkl is not trained/present.")

        # High credit history, high income - likely approved
        form_data = {
            "Gender": "Male",
            "Married": "Yes",
            "Dependents": "0",
            "Education": "Graduate",
            "Self_Employed": "No",
            "ApplicantIncome": "8000",
            "CoapplicantIncome": "2000",
            "LoanAmount": "100",
            "Loan_Amount_Term": "360",
            "Credit_History": "1",
            "Property_Area": "Semiurban"
        }

        response = self.client.post('/predict', data=form_data)
        self.assertEqual(response.status_code, 200)
        
        # Verify JSON response
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['status'], 'success')
        self.assertEqual(data['redirect'], '/analyzing')

        # Verify that session stores the prediction results
        with self.client.session_transaction() as sess:
            self.assertIn('loan_status', sess)
            self.assertIn('score', sess)
            self.assertIn('emi', sess)
            self.assertIn('risk_level', sess)
            self.assertEqual(sess['gender'], 'Male')
            self.assertEqual(sess['applicant_income'], 8000.0)

        # Now GET results page with session active
        results_response = self.client.get('/results')
        self.assertEqual(results_response.status_code, 200)
        self.assertIn(b'Your Eligibility Results', results_response.data)

    def test_prediction_rejection_profile(self):
        """Test POST /predict with a high-risk profile and check rejection characteristics."""
        if model_data is None:
            self.skipTest("Skipping prediction test because best_model.pkl is not trained/present.")

        # Low income, high loan, bad credit history - likely rejected
        form_data = {
            "Gender": "Female",
            "Married": "No",
            "Dependents": "3+",
            "Education": "Not Graduate",
            "Self_Employed": "Yes",
            "ApplicantIncome": "1500",
            "CoapplicantIncome": "0",
            "LoanAmount": "400",
            "Loan_Amount_Term": "180",
            "Credit_History": "0",
            "Property_Area": "Rural"
        }

        response = self.client.post('/predict', data=form_data)
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['status'], 'success')

        with self.client.session_transaction() as sess:
            self.assertEqual(sess['loan_status'], 'Rejected')
            self.assertEqual(sess['risk_level'], 'High Risk')
            self.assertEqual(sess['credit_history'], 0.0)

    def test_prediction_invalid_input(self):
        """Test POST /predict with invalid data formats causing exceptions."""
        form_data = {
            "Gender": "Male",
            "ApplicantIncome": "not-a-number",  # triggers ValueError in float conversion
            "LoanAmount": "100"
        }
        response = self.client.post('/predict', data=form_data)
        self.assertEqual(response.status_code, 500)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(data['status'], 'error')

if __name__ == '__main__':
    unittest.main()
