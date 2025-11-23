"""
Test functions from codependence module.
"""

import unittest
import numpy as np
import pandas as pd

from arbitragelab.codependence.correlation import (squared_angular_distance, angular_distance, absolute_angular_distance,
                                                   distance_correlation, kullback_leibler_distance, norm_distance)
from arbitragelab.codependence.information import (get_mutual_info, variation_of_information_score,
                                                   get_optimal_number_of_bins)
from arbitragelab.codependence.codependence_matrix import (get_dependence_matrix, get_distance_matrix)

# pylint: disable=invalid-name

class TestCodependence(unittest.TestCase):
    """
    Test codependence module.
    """

    def setUp(self):
        """
        Set the file path for the sample dollar bars data.
        """
        state = np.random.RandomState(42)
        self.x = state.normal(size=1000)
        self.y_1 = self.x ** 2 + state.normal(size=1000) / 5
        self.y_2 = abs(self.x) + state.normal(size=1000) / 5
        self.y_3 = self.x + state.normal(size=1000) / 5
        # Create a simple test matrix instead of using get_classification_data
        self.X_matrix = pd.DataFrame(state.normal(size=(100, 6)), columns=[f'feature_{i}' for i in range(6)])
        # Setting sample correlation matrices
        self.corr_A = np.array([[1, 0.70573243, 0.03085437, 0.6019651, 0.81214341],
                                [0.70573243, 1, 0.03126594, 0.56559443, 0.88961155],
                                [0.03085437, 0.03126594, 1, 0.01760481, 0.02842086],
                                [0.60196510, 0.56559443, 0.01760481, 1, 0.73827921],
                                [0.81214341, 0.88961155, 0.02842086, 0.73827921, 1]])
        self.corr_B = np.array([[1, 0.49805826, 0.00095199, 0.36236198, 0.65957691],
                                [0.49805826, 1, 0.00097755, 0.31989705, 0.79140871],
                                [0.00095199, 0.00097755, 1, 0.00030992, 0.00080774],
                                [0.36236198, 0.31989705, 0.00030992, 1, 0.54505619],
                                [0.65957691, 0.79140871, 0.00080774, 0.54505619, 1]])

    def test_correlations(self):
        """
        Test correlation based coefficients: angular (abs, square), distance correlation.
        """
        angular_dist = angular_distance(self.x, self.y_1)
        sq_angular_dist = squared_angular_distance(self.x, self.y_1)
        abs_angular_dist = absolute_angular_distance(self.x, self.y_1)
        dist_corr = distance_correlation(self.x, self.y_1)
        kullback_dist = kullback_leibler_distance(self.corr_A, self.corr_B)
        norm_dist = norm_distance(self.corr_A, self.corr_B)
        # Assigns pd.DataFrame as input
        corr_A_df = pd.DataFrame(self.corr_A)
        corr_B_df = pd.DataFrame(self.corr_B)
        kullback_dist_df = kullback_leibler_distance(corr_A_df, corr_B_df)
        norm_dist_df = norm_distance(corr_A_df, corr_B_df)

        self.assertAlmostEqual(angular_dist, 0.67, delta=1e-2)
        self.assertAlmostEqual(abs_angular_dist, 0.6703, delta=1e-2)
        self.assertAlmostEqual(sq_angular_dist, 0.7, delta=1e-2)
        self.assertAlmostEqual(dist_corr, 0.529, delta=1e-2)
        self.assertAlmostEqual(kullback_dist, 0.250807852409, delta=1e-2)
        self.assertAlmostEqual(norm_dist, 0.58255075616, delta=1e-2)

        # Checking if return is consistent when input type is pd.DataFrame
        self.assertAlmostEqual(kullback_dist_df, kullback_dist, delta=1e-2)
        self.assertAlmostEqual(norm_dist_df, norm_dist, delta=1e-2)

        dist_corr_y_2 = distance_correlation(self.x, self.y_2)
        self.assertAlmostEqual(dist_corr_y_2, 0.5216, delta=1e-2)

    def test_information_metrics(self):
        """
        Test mutual info, information variability metrics.
        """
        # Test mutual info score
        mut_info = get_mutual_info(self.x, self.y_1, normalize=False)
        mut_info_norm = get_mutual_info(self.x, self.y_1, normalize=True)
        mut_info_bins = get_mutual_info(self.x, self.y_1, n_bins=10)
        mut_info_stand_copula = get_mutual_info(self.x, self.y_1, normalize=False, estimator='standard_copula')
        mut_info_copula_entropy = get_mutual_info(self.x, self.y_1, normalize=False, estimator='copula_entropy')
        mut_info_stand_copula_norm = get_mutual_info(self.x, self.y_1, normalize=True, estimator='standard_copula')
        mut_info_copula_entropy_norm = get_mutual_info(self.x, self.y_1, normalize=True, estimator='copula_entropy')

        self.assertAlmostEqual(mut_info, 0.522, delta=1e-2)
        self.assertAlmostEqual(mut_info_norm, 0.64, delta=1e-2)
        self.assertAlmostEqual(mut_info_bins, 0.626, delta=1e-2)
        self.assertAlmostEqual(mut_info_stand_copula, 0.876, delta=1e-2)
        self.assertAlmostEqual(mut_info_copula_entropy, 0.879, delta=1e-2)
        self.assertAlmostEqual(mut_info_stand_copula_norm, 0.399, delta=1e-2)
        self.assertAlmostEqual(mut_info_copula_entropy_norm, 0.400, delta=1e-2)

        # Test information variation score
        info_var = variation_of_information_score(self.x, self.y_1, normalize=False)
        info_var_norm = variation_of_information_score(self.x, self.y_1, normalize=True)
        info_var_bins = variation_of_information_score(self.x, self.y_1, n_bins=10)

        self.assertAlmostEqual(info_var, 1.4256, delta=1e-2)
        self.assertAlmostEqual(info_var_norm, 0.7316, delta=1e-2)
        self.assertAlmostEqual(info_var_bins, 1.418, delta=1e-2)

    def test_number_of_bins(self):
        """
        Test get_optimal_number_of_bins functions.
        """

        n_bins_x = get_optimal_number_of_bins(self.x.shape[0])
        n_bins_x_y = get_optimal_number_of_bins(self.x.shape[0], np.corrcoef(self.x, self.y_1)[0, 1])

        self.assertEqual(n_bins_x, 15)
        self.assertEqual(n_bins_x_y, 9)

    def test_codependence_matrix(self):
        '''
        Test the get_dependence_matrix and get_distance_matrix function
        '''

        # TODO: add tests for values in matrix
        #Dependence_matrix

        vi_matrix = get_dependence_matrix(self.X_matrix, dependence_method='information_variation')
        mi_matrix = get_dependence_matrix(self.X_matrix, dependence_method='mutual_information')
        corr_matrix = get_dependence_matrix(self.X_matrix, dependence_method='distance_correlation')

        #Distance_matrix
        angl = get_distance_matrix(vi_matrix, distance_metric='angular')
        sq_angl = get_distance_matrix(mi_matrix, distance_metric='squared_angular')
        abs_angl = get_distance_matrix(corr_matrix, distance_metric='abs_angular')

        #assertions
        self.assertEqual(vi_matrix.shape[0], self.X_matrix.shape[1])
        self.assertEqual(mi_matrix.shape[0], self.X_matrix.shape[1])
        self.assertEqual(corr_matrix.shape[0], self.X_matrix.shape[1])

        self.assertEqual(angl.shape[0], self.X_matrix.shape[1])
        self.assertEqual(sq_angl.shape[0], self.X_matrix.shape[1])
        self.assertEqual(abs_angl.shape[0], self.X_matrix.shape[1])

    def test_value_error_raise(self):
        '''
        Test of invailid arguments
        '''
        #Unkown dependence_metric
        with self.assertRaises(ValueError):
            get_dependence_matrix(self.X_matrix, dependence_method='unknown')
        #Unkown distance_metric
        with self.assertRaises(ValueError):
            get_distance_matrix(self.X_matrix, distance_metric='unknown')

