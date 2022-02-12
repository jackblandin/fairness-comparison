from BlackBoxAuditing.repairers.GeneralRepairer import Repairer
from pandas import DataFrame
from fairness.algorithms.Algorithm import Algorithm
from fairness.algorithms.baseline.Generic import Generic

REPAIR_LEVEL_DEFAULT = 1.0

class BlandinAlgorithm(Generic):
    def __init__(self, model):
        Generic.__init__(self)
        self.classifier = model.get_classifier()
        self.name = f'Blandin-{model.name}'
