from unittest import TestCase
import timesynth as ts


class TestGaussianNoise(TestCase):
    def setup(self):
        print('Setup')

    def test_sample_next(self):
        self.fail()

    def test_sample_vectorized(self):
        self.fail()

    def run_test(self):
        time_sampler = ts.TimeSampler(stop_time=20)
        irregular_time_samples = time_sampler._sample_irregular_time(
            num_points=500, keep_percentage=50
        )
        white_noise = ts.noise.GaussianNoise(std=0.3)
        wnoise_vec = white_noise.sample_vectorized(irregular_time_samples)
        wnoise_value = white_noise.sample_next(irregular_time_samples[0], None, None)
        return wnoise_vec, wnoise_value

    def test_gaussian_noise(self):
        wnoise_vec, wnoise_value = self.run_test()
        self.assertEqual(len(wnoise_vec), 250, "Should be 250")

    if __name__ == '__main__':
        unittest.main()