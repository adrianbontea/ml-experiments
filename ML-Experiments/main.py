import asyncio
from californiahousing.dataanalysis import DataAnalysisExperiment
from californiahousing.preprocessing import PreProcessingExperiment
import dependency_injector.containers as containers
import dependency_injector.providers as providers

class ExperimentRunner:
    def __init__(self, experiment):
        self.__experiment = experiment

    async def run_async(self):
        await self.__experiment.run_async()

class ExperimentsContainer(containers.DeclarativeContainer):
    data_analysis = providers.Factory(DataAnalysisExperiment)
    preprocessing = providers.Factory(PreProcessingExperiment)

class RunnersContainer(containers.DeclarativeContainer):
    instance = providers.Factory(ExperimentRunner, experiment = ExperimentsContainer.data_analysis)

async def main():
    runner = RunnersContainer.instance()
    await runner.run_async()

asyncio.run(main())



