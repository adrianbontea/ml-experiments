from abc import ABC, abstractmethod
import asyncio

class ExperimentBase(ABC):
    async def run_async(self):
        await asyncio.create_task(self.do_run_async())

    @abstractmethod
    async def do_run_async(self):
        pass


