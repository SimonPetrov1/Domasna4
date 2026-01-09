from abc import ABC, abstractmethod

class AnalysisStrategy(ABC):
    """
    Strategy interface for all analysis types.
    """

    @abstractmethod
    def analyze(self):
        pass