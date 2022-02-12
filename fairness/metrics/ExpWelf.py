import sys
import numpy as np
import math

from fairness.metrics.utils import calc_pos_protected_percents
from fairness.metrics.UtilityMetric import UtilityMetric

class ExpWelf(UtilityMetric):

    def __init__(self, welfare_fn, cost_fn):
        UtilityMetric.__init__(self, welfare_fn, cost_fn)
        self.name = f'ExpWelf_{welfare_fn.__name__}'

    def calc(self, actual, predicted, dict_of_sensitive_lists, single_sensitive_name,
            unprotected_vals, positive_pred, dict_of_nonclass_attrs):

        # Compute welfare for each sensitive class
        prot_welf, unprot_welf = self.calc_welfare(actual,
                                                   predicted,
                                                   dict_of_sensitive_lists,
                                                   single_sensitive_name,
                                                   unprotected_vals,
                                                   positive_pred,
                                                   dict_of_nonclass_attrs)

        comb_welf = np.concatenate([prot_welf, unprot_welf])
        exp_welf = np.mean(comb_welf)
        print(f'exp_welf: {exp_welf}')

        return exp_welf
