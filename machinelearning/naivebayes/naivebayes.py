import numpy as np
import os
import pandas


class NaiveBayes:
    """A naive Bayes classifier.

    Classes of observations are always assumed to be in the final column of the dataset.

    """

    def __init__(self, dataset='', headerPresent=False, separator='\t'):
        if type(dataset) == str:
            if os.path.isfile(dataset):
                self.dataset = pandas.read_csv(dataset, sep=separator, header=(0 if headerPresent else None))
            else:
                self.dataset = pandas.DataFrame()
        else:
            self.dataset = pandas.DataFrame(dataset)

        self.numberOfObservations = len(self.dataset.index)  # Determine the total number of observations.
        self.groupedByClasses = self.dataset.groupby(self.dataset.iloc[:,-1])  # Group observations by classes.
        self.classes = [i for i in self.groupedByClasses.grouper.groups]
        self.features = self.dataset.columns[:-1]

        # Compute the class priors.
        self.classPriors = (self.groupedByClasses.size()) / self.numberOfObservations

    def predict(self, dataset, likelihoodDistribution='Gaussian'):
        """
        """
        
        predictedClasses = []

        if likelihoodDistribution == 'Gaussian':
            self.parameters = self.groupedByClasses.aggregate([np.mean, np.var])[self.features]
            print(self.parameters)
            print(self.classes)
            print(self.features)
            for i in self.classes:
                print('\n',i,'\n',self.parameters.xs(i))