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

from lab_8_sft.service import app


class WebServiceTest(unittest.TestCase):
    """
    Tests web service
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls._app = app

        cls._client = TestClient(app)

    @pytest.mark.lab_8_sft
    @pytest.mark.mark10
    def test_e2e_ideal(self) -> None:
        """
        Ideal service scenario
        """

        url = "/infer"
        input_text = "What is the capital of France?"

        for model_type in range(2):
            payload = {"question": input_text, "is_base_model": bool(model_type)}
            response = self._client.post(url, json=payload)

            self.assertEqual(200, response.status_code)
            self.assertIn("infer", response.json())
            print(response.json().get("infer"))
            self.assertIsNotNone(response.json().get("infer"))
