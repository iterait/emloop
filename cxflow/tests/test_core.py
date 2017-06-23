"""
Test module with base cxflow test case classes.
"""
import logging
import shutil
import tempfile
from unittest import TestCase


class CXTestCase(TestCase):
    """Base cxflow test case which disables logging."""

    def __init__(self, *args, **kwargs):
        """Create a new test case and disable logging."""
        logging.getLogger().disabled = True
        super().__init__(*args, **kwargs)


class CXTestCaseWithDir(CXTestCase):
    """Cxflow test case with temp dir available."""

    def __init__(self, *args, **kwargs):
        """Create a new test case."""
        self.tmpdir = None
        super().__init__(*args, **kwargs)

    def setUp(self):
        """Create a temp dir before every test method."""
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        """Remove the respective temp dir after every test method."""
        shutil.rmtree(self.tmpdir)
