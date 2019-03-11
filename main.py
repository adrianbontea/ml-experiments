import asyncio
from californiahousing.dataanalysis import DataAnalysisExperiment
from californiahousing.preprocessing import PreProcessingExperiment
from californiahousing.models import ModelsExperiment
from classification.binary import BinaryClassificationExperiment
from classification.multiclass import MulticlassClassificationExperiment
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
    binary_classification = providers.Factory(BinaryClassificationExperiment)
    multiclass_classification = providers.Factory(MulticlassClassificationExperiment)


class RunnersContainer(containers.DeclarativeContainer):
    instance = providers.Factory(ExperimentRunner, experiment=ExperimentsContainer.multiclass_classification)


async def main():
    runner = RunnersContainer.instance()
    await runner.run_async()


asyncio.run(main())
