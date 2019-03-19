import asyncio
from californiahousing import *
from classification import *
from whitebox import *
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
    multilabel_classification = providers.Factory(MultilabelClassificationExperiment)
    multioutput_classification = providers.Factory(MultioutputClassificationExperiment)
    whitebox_linear_regression = providers.Factory(LinearRegressionExperiment)
    whitebox_polynomial_regression = providers.Factory(PolynomialRegressionExperiment)


class RunnersContainer(containers.DeclarativeContainer):
    instance = providers.Factory(ExperimentRunner, experiment=ExperimentsContainer.whitebox_polynomial_regression)


async def main():
    runner = RunnersContainer.instance()
    await runner.run_async()


asyncio.run(main())
