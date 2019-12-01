# import the necessary modules needed
# for testing the predictions file

from app import app 
import unittest 

# Initialize the class and then specify
# the Tear up and down method which would
# Actuate the tests based on core_demand

class PredictionTest(unittest.TestCase):

    # Define the setup class to initialize the
    # App class as a test Object during testing

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    # Define the teardown class to remove the 
    # App class as a test Object during testing

    def tearDown(self):
        pass 

    # Test that the homepage opens correctly on
    # Accessing the Prediction page via_urls

    def test_homepage_opens_correctly(self):
        response = self.app.get('/')
        self.assertEquals(response.status_code, 200)

    # Test that the result page is opened correctly
    # And that it has a POST request sent to it

    def test_prediction_processes_post(self):
        response = self.app.get('/result')
        self.assertEquals(response.status_code, 200)
        self.assertIsNot(response.status_code, 401)
        