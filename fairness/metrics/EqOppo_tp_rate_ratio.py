"""Equal opportunity - Protected and unprotected True Positive Rate ratio"""
import math
import sys
import numpy as np

from fairness.metrics.utils import calc_fp_fn, calc_tp_tn
from fairness.metrics.Metric import Metric

class EqOppo_tp_rate_ratio(Metric):
    def __init__(self):
        Metric.__init__(self)
        self.name = 'EqOppo_tp_rate_ratio'

    def calc(self, actual, predicted, dict_of_sensitive_lists, single_sensitive_name,
             unprotected_vals, positive_pred, dict_of_nonclass_attrs):
        sensitive = dict_of_sensitive_lists[single_sensitive_name]

        tp_unprotected, tp_protected, tn_protected, tn_unprotected = \
        calc_tp_tn(actual, predicted, sensitive, unprotected_vals, positive_pred)

        fp_unprotected,fp_protected, fn_protected, fn_unprotected = \
        calc_fp_fn(actual, predicted, sensitive, unprotected_vals, positive_pred)

        tp_unprotected_rate = tp_unprotected / (tp_unprotected + fn_unprotected)
        tp_protected_rate = tp_protected / (tp_protected + fn_protected)

        tp_ratio=0.0
        if tp_unprotected_rate > 0:
            tp_ratio = tp_protected_rate / tp_unprotected_rate
        if tp_unprotected_rate == 0.0 and tp_protected_rate == 0.0:
            tp_ratio = 1.0

        return tp_ratio
