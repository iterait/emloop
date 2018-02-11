from cxflow.models.ensemble_model import major_vote, EnsembleModel

from ..test_core import CXTestCase


class MajorVoteTest(CXTestCase):
    """major_vote function test case."""

    def test_major_vote(self):
        """Test if majo_vote works properly."""
        vote1 = (1, 3, (5, 5), 12)
        vote2 = [1, 2, 2, 2]
        vote3 = [1, 2, (5, 5), 1]

        result = major_vote([vote1, vote2, vote3])
        self.assertListEqual([1, 2, (5, 5)], list(result)[:3])
        self.assertIn(result[-1], {1, 2, 12})

