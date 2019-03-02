import asyncio
from californiahousing.dataanalysis import DataAnalysisExperiment
from californiahousing.preprocessing import PreProcessingExperiment
from californiahousing.models import ModelsExperiment
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
    models = providers.Factory(ModelsExperiment)

class RunnersContainer(containers.DeclarativeContainer):
    instance = providers.Factory(ExperimentRunner, experiment = ExperimentsContainer.models)

async def main():
    runner = RunnersContainer.instance()
    await runner.run_async()

asyncio.run(main())



