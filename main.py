import asyncio
from californiahousing.dataanalysis import DataAnalysisExperiment
from californiahousing.preprocessing import PreProcessingExperiment
from californiahousing.models import ModelsExperiment
from classification.binary import BinaryClassificationExperiment
from classification.multiclass import MulticlassClassificationExperiment
from classification.multilabel import MultilabelClassificationExperiment
from classification.multioutput import MultioutputClassificationExperiment
from whitebox.linearregression import LinearRegressionExperiment
from whitebox.polynomialregression import PolynomialRegressionExperiment
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
