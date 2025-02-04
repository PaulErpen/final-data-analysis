import unittest
import numpy as np
from pwrc import compute_pwrc


class PwrcTest(unittest.TestCase):
    def test_given_a_random_assortment_of_MOS_scores_and_ssim_values__when_computing__then_do_not_throw_an_error(
        self,
    ) -> None:
        N = 10
        ssim = np.random.normal(size=N)
        MOS = np.random.randint(0, 5, N)

        compute_pwrc(ssim, MOS)

    def test_given_a_random_assortment_of_MOS_scores_and_ssim_values__when_computing__then_return_a_float(
        self,
    ) -> None:
        N = 10
        ssim = np.random.normal(size=N)
        MOS = np.random.randint(0, 5, N)

        return_val = compute_pwrc(ssim, MOS)

        self.assertIsInstance(return_val, float)

    def test_given_a_random_assortment_of_MOS_scores_and_ssim_values__when_computing__then_the_value_must_be_within_0_and_1(
        self,
    ) -> None:
        N = 10
        ssim = np.random.normal(size=N)
        MOS = np.random.randint(0, 5, N)

        return_val = compute_pwrc(ssim, MOS)

        self.assertLessEqual(return_val, 1)
        self.assertGreaterEqual(return_val, 0)
