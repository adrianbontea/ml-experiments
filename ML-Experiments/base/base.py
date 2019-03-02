from abc import ABC, abstractmethod

class ExperimentBase(ABC):
    @abstractmethod
    async def run_async(self):
        pass


