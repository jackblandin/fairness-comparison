""" Equal opportunity - Protected and unprotected False postives rate ratio"""
import math
import sys
import numpy as np

from fairness.metrics.utils import calc_fp_fn, calc_tp_tn
from fairness.metrics.Metric import Metric

class EqOppo_fp_rate_ratio(Metric):
    def __init__(self):
        Metric.__init__(self)
        self.name = 'EqOppo_fp_rate_ratio'

    def calc(self, actual, predicted, dict_of_sensitive_lists, single_sensitive_name,
             unprotected_vals, positive_pred, dict_of_nonclass_attrs):
        sensitive = dict_of_sensitive_lists[single_sensitive_name]

        fp_unprotected,fp_protected, fn_protected, fn_unprotected = \
        calc_fp_fn(actual, predicted, sensitive, unprotected_vals, positive_pred)

        tp_unprotected, tp_protected, tn_protected, tn_unprotected = \
        calc_tp_tn(actual, predicted, sensitive, unprotected_vals, positive_pred)

        fp_unprotected_rate = fp_unprotected / (fp_unprotected + tn_unprotected)
        fp_protected_rate = fp_protected / (fp_protected + tn_protected)

        fp_ratio=0.0
        if fp_unprotected > 0:
            fp_ratio = fp_protected_rate / fp_unprotected_rate
        if fp_unprotected == 0.0 and fp_protected == 0.0:
            fp_ratio=1.0

        return fp_ratio
