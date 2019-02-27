import multiprocessing

from californiahousing.dataanalysis import DataAnalysisExperiment

exp = DataAnalysisExperiment()

if __name__ == '__main__':
    data_analysis_process = multiprocessing.Process(target=exp.run)
    data_analysis_process.start()



