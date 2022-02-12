"""Equal opportunity - Protected and unprotected False negative ratio"""
import math
import sys
import numpy as np

from fairness.metrics.utils import calc_fp_fn
from fairness.metrics.Metric import Metric

class EqOppo_fn_ratio(Metric):
    def __init__(self):
        Metric.__init__(self)
        self.name = 'EqOppo_fn_ratio'

    def calc(self, actual, predicted, dict_of_sensitive_lists, single_sensitive_name,
             unprotected_vals, positive_pred, dict_of_nonclass_attrs):
        sensitive = dict_of_sensitive_lists[single_sensitive_name]

        fp_unprotected,fp_protected, fn_protected, fn_unprotected = \
        calc_fp_fn(actual, predicted, sensitive, unprotected_vals, positive_pred)

        # JDB: modified to compute the ratio of the RATES, not the raw counts
        unprotected = np.sum([s in unprotected_vals for s in sensitive])
        protected = np.sum([s not in unprotected_vals for s in sensitive])

        fn_unprotected_rate = fn_unprotected / unprotected
        fn_protected_rate = fn_protected / protected

        fn_ratio=0.0
        if fn_unprotected_rate > 0:
            fn_ratio = fn_protected_rate / fn_unprotected_rate
        if fn_unprotected_rate == 0.0 and fn_protected_rate == 0.0:
            fn_ratio = 1.0
        return fn_ratio

