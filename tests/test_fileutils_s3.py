import importlib
import os
import sys
import types
import unittest


class TestS3EndpointUrl(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._original_s3_hostname = os.environ.get("S3_HOSTNAME")
        cls._original_modules = {}
        for module_name in ("numpy", "eccodes", "fsspec", "pyproj"):
            cls._original_modules[module_name] = sys.modules.get(module_name)
            if module_name not in sys.modules:
                sys.modules[module_name] = types.ModuleType(module_name)

        if "fileutils" in sys.modules:
            cls.fileutils = importlib.reload(sys.modules["fileutils"])
        else:
            cls.fileutils = importlib.import_module("fileutils")

    @classmethod
    def tearDownClass(cls):
        if cls._original_s3_hostname is None:
            os.environ.pop("S3_HOSTNAME", None)
        else:
            os.environ["S3_HOSTNAME"] = cls._original_s3_hostname

        for module_name, original_module in cls._original_modules.items():
            if original_module is None:
                del sys.modules[module_name]
            else:
                sys.modules[module_name] = original_module

    def setUp(self):
        os.environ.pop("S3_HOSTNAME", None)

    def test_default_s3_host_is_used_when_env_is_missing(self):
        self.assertEqual(
            self.fileutils.get_s3_endpoint_url(),
            "https://lake.fmi.fi",
        )

    def test_s3_host_without_protocol_gets_https_prefix(self):
        os.environ["S3_HOSTNAME"] = "netapp.example"
        self.assertEqual(
            self.fileutils.get_s3_endpoint_url(),
            "https://netapp.example",
        )

    def test_s3_host_with_protocol_is_used_as_is(self):
        os.environ["S3_HOSTNAME"] = "https://custom.example"
        self.assertEqual(
            self.fileutils.get_s3_endpoint_url(),
            "https://custom.example",
        )


if __name__ == "__main__":
    unittest.main()
