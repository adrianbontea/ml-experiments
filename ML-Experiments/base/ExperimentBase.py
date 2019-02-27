from abc import ABC, abstractmethod

class ExperimentBase(ABC):
    @abstractmethod
    def run(self):
        pass


