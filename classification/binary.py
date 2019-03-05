from classification.base import ClassificationExperimentBase


class BinaryClassificationExperiment(ClassificationExperimentBase):
    async def do_run_async(self):
        training_set = super().load_train_images()
        training_labels = super().load_train_labels()
        test_set = super().load_test_images()
        test_labels = super().load_test_labels()
