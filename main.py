import asyncio
from californiahousing import *
from classification import *
from whitebox import *
from svm import *
from decisiontrees import *
from ensemble import *
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
    whitebox_logistic_regression = providers.Factory(LogisticRegressionExperiment)
    svm_linear_classification = providers.Factory(LinearSvmClassificationExperiment)
    svm_non_linear_classification = providers.Factory(NonLinearSvmClassificationExperiment)
    svm_linear_regression = providers.Factory(LinearSvmRegressionExperiment)
    svm_non_linear_regression = providers.Factory(NonLinearSvmRegressionExperiment)
    decision_tree_visualization = providers.Factory(DecisionTreeVisualizationExperiment)
    decision_tree_regularization = providers.Factory(DecisionTreeRegularizationExperiment)
    decision_tree_regression = providers.Factory(DecisionTreeRegressionExperiment)
    voting_classifier = providers.Factory(VotingClassifierExperiment)
    bagging_pasting = providers.Factory(BaggingAndPastingExperiment)


class RunnersContainer(containers.DeclarativeContainer):
    instance = providers.Factory(ExperimentRunner, experiment=ExperimentsContainer.bagging_pasting)


async def main():
    runner = RunnersContainer.instance()
    await runner.run_async()


asyncio.run(main())
