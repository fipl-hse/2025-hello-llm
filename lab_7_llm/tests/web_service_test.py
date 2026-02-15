"""
Checks that the service is working properly
"""

# pylint: disable=duplicate-code
import unittest
from collections import namedtuple

import pytest

try:
    from fastapi.testclient import TestClient
except ImportError:
    print('Library "fastapi" not installed. Failed to import.')
    TestClient = namedtuple("TestClient", "post")

from lab_7_llm.service import app


class WebServiceTest(unittest.TestCase):
    """
    Tests web service
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls._app = app

        if app is not None:
            cls._client = TestClient(app)

    @pytest.mark.lab_7_llm
    @pytest.mark.mark10
    def test_e2e_ideal(self) -> None:
        """
        Ideal service scenario
        """
        url = "/infer"
        input_text = "What is the capital of France?"
        input_context = ("Everybody knows that Paris is the "
                         "biggest and the most popular place in France, "
                         "it is obvious that it is the capital")

        payload = {"question": input_text, "context": input_context}
        response = self._client.post(url, json=payload)

        self.assertEqual(200, response.status_code)
        self.assertIn("infer", response.json())
        print(response.json().get("infer"))
        self.assertIsNotNone(response.json().get("infer"))
