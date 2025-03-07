import numpy

from fairness.metrics.Accuracy import Accuracy
from fairness.metrics.BCR import BCR
from fairness.metrics.CalibrationNeg import CalibrationNeg
from fairness.metrics.CalibrationPos import CalibrationPos
from fairness.metrics.CV import CV
from fairness.metrics.CVWelf import CVWelf
from fairness.metrics.DemParWelf import DemParWelf
from fairness.metrics.DIAvgAll import DIAvgAll
from fairness.metrics.DIBinary import DIBinary
from fairness.metrics.EqOppo_fn_diff import EqOppo_fn_diff
from fairness.metrics.EqOppo_fn_ratio import EqOppo_fn_ratio
from fairness.metrics.EqOppo_fp_diff import EqOppo_fp_diff
from fairness.metrics.EqOppo_fp_ratio import EqOppo_fp_ratio
from fairness.metrics.EqOppo_fp_rate_ratio import EqOppo_fp_rate_ratio
from fairness.metrics.EqOppo_fp_ratio import EqOppo_fp_ratio
from fairness.metrics.EqOppo_tp_rate_ratio import EqOppo_tp_rate_ratio
from fairness.metrics.ExpCost import ExpCost
from fairness.metrics.ExpWelf import ExpWelf
from fairness.metrics.FNR import FNR
from fairness.metrics.FPR import FPR
from fairness.metrics.MCC import MCC
from fairness.metrics.SensitiveMetric import SensitiveMetric
from fairness.metrics.TNR import TNR
from fairness.metrics.TPR import TPR

from fairness.metrics.UtilityMetric import pos_rate, accuracy, german_credit_payoffs


METRICS = [
        # accuracy metrics
        Accuracy(),
        # TPR(),
        # TNR(),
        # BCR(),
        # MCC(),
        # fairness metrics
        DIBinary(),
        # DIAvgAll(),
        # CV(),
        EqOppo_tp_rate_ratio(),
        EqOppo_fp_rate_ratio(),
        # DemParWelf(welfare_fn=pos_rate, cost_fn=pos_rate),
        # DemParWelf(welfare_fn=accuracy, cost_fn=accuracy),
        DemParWelf(welfare_fn=german_credit_payoffs, cost_fn=None, transform_welf=(lambda w: -1*w)),
        DemParWelf(welfare_fn=german_credit_payoffs, cost_fn=None, transform_welf=(lambda w: -1*(w**2)), name='german_credit_payoffs_imbal'),
        DemParWelf(welfare_fn=german_credit_payoffs, cost_fn=None, tau=-1, transform_welf=(lambda w: -1*w)),
        DemParWelf(welfare_fn=german_credit_payoffs, cost_fn=None, tau=-1, transform_welf=(lambda w: -1*(w**2)), name='german_credit_payoffs_imbal'),
        DemParWelf(welfare_fn=german_credit_payoffs, cost_fn=None, tau=1, transform_welf=(lambda w: 5+(-1*w)), name='german_credit_payoffs_shifted_pos'),
        CVWelf(welfare_fn=german_credit_payoffs, cost_fn=None, transform_welf=(lambda w: -1*w)),
        CVWelf(welfare_fn=german_credit_payoffs, cost_fn=None, transform_welf=(lambda w: -1*(w**4)), name='german_credit_payoffs_imbal'),
        # ExpWelf(welfare_fn=pos_rate, cost_fn=None),
        # ExpWelf(welfare_fn=accuracy, cost_fn=None),
        ExpWelf(welfare_fn=german_credit_payoffs, cost_fn=None),
        ExpCost(welfare_fn=None, cost_fn=german_credit_payoffs),
        # other fairness metrics
        # SensitiveMetric(Accuracy),
        # SensitiveMetric(TPR),
        # SensitiveMetric(TNR),
        # SensitiveMetric(FPR),
        # SensitiveMetric(FNR),
        SensitiveMetric(BCR),
        # SensitiveMetric(CalibrationPos),
        # SensitiveMetric(CalibrationNeg),
        ]

def get_metrics(dataset, sensitive_dict, tag):
    """
    Takes a dataset object and a dictionary mapping sensitive attributes to a list of the sensitive
    values seen in the data.  Returns an expanded list of metrics based on the base METRICS.
    """
    metrics = []
    for metric in METRICS:
        metrics += metric.expand_per_dataset(dataset, sensitive_dict, tag)
    return metrics

def add_metric(metric):
    METRICS.append(metric)
