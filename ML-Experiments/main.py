import gc

from californiahousing.dataanalysis import DataAnalysisExperiment

exp = DataAnalysisExperiment()
exp.run()

exp = None
gc.collect()


