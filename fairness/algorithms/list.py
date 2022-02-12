from fairness.algorithms.zafar.ZafarAlgorithm import ZafarAlgorithmBaseline, ZafarAlgorithmAccuracy, ZafarAlgorithmFairness
from fairness.algorithms.kamishima.KamishimaAlgorithm import KamishimaAlgorithm
from fairness.algorithms.kamishima.CaldersAlgorithm import CaldersAlgorithm
from fairness.algorithms.feldman.FeldmanAlgorithm import FeldmanAlgorithm
from fairness.algorithms.blandin.BlandinAlgorithm import BlandinAlgorithm
from fairness.algorithms.baseline.Random import Random
from fairness.algorithms.baseline.SVM import SVM
from fairness.algorithms.baseline.DecisionTree import DecisionTree
from fairness.algorithms.baseline.GaussianNB import GaussianNB
from fairness.algorithms.baseline.LogisticRegression import LogisticRegression
from fairness.algorithms.ParamGridSearch import ParamGridSearch
from fairness.algorithms.Ben.SDBSVM import SDBSVM

from fairness.metrics.DIAvgAll import DIAvgAll
from fairness.metrics.Accuracy import Accuracy
from fairness.metrics.MCC import MCC


ALGORITHMS = [
   ## baseline
   # Random(),
   SVM(),
   # # GaussianNB(),
   LogisticRegression(),
   DecisionTree(),
   # KamishimaAlgorithm(),                                          # Kamishima
   # CaldersAlgorithm(),                                            # Calders
   # ZafarAlgorithmBaseline(),                                      # Zafar
   ZafarAlgorithmFairness(),
   # ZafarAlgorithmAccuracy(),
# #   SDBSVM(),                                                      # not yet confirmed to work
   # # ParamGridSearch(KamishimaAlgorithm(), Accuracy()),             # Kamishima params
   # # ParamGridSearch(KamishimaAlgorithm(), DIAvgAll()),
   FeldmanAlgorithm(SVM()),
   # # FeldmanAlgorithm(GaussianNB()),       # Feldman
   FeldmanAlgorithm(LogisticRegression()),
   FeldmanAlgorithm(DecisionTree()),
   # BlandinAlgorithm(GaussianNB()),                                  # Blandin
   # # BlandinAlgorithm(LogisticRegression()),
   # # ParamGridSearch(FeldmanAlgorithm(SVM()), DIAvgAll()),          # Feldman params
   # # ParamGridSearch(FeldmanAlgorithm(SVM()), Accuracy()),
   # # ParamGridSearch(FeldmanAlgorithm(GaussianNB()), DIAvgAll()),
   # # ParamGridSearch(FeldmanAlgorithm(GaussianNB()), Accuracy()),
]

def add_algorithm(algorithm):
    ALGORITHMS.append(algorithm)
