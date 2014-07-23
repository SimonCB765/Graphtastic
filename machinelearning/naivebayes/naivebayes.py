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

        # Maximum likelihood computation of class priors.
        self.classPriors = (self.groupedByClasses.size()) / self.numberOfObservations

    def predict(self, dataset, likelihoodDistribution='Gaussian'):
        """
        """
        
        predictedClasses = []

        if likelihoodDistribution == 'Gaussian':
            self.parameters = self.groupedByClasses.aggregate([np.mean, np.var])[self.features]
            classMeanVectors = {}
            classVarianceVectors = {}
            # Determine class mean and variance vectors for the observations in the class.
            for i in self.classes:
                classParameters = self.parameters.xs(i)
                classMeanVector = classParameters.xs('mean', level=1)
                classMeanVectors[i] = classMeanVector
                classVarianceVector = classParameters.xs('var', level=1)
                classVarianceVectors[i] = classVarianceVector

            for index, series in list(dataset.iterrows()):
                classPosteriors = []
                for j in self.classes:
                    classPrior = self.classPriors.xs(j)
                    classMean = classMeanVectors[j]
                    classVariance = classVarianceVectors[j]
                    classLikelihood = (1 / np.sqrt(2 * np.pi * classVariance)) * np.exp(- np.square(series - classMean) / (2 * classVariance))
                    posterior = classPrior * classLikelihood.prod()
                    classPosteriors.append(posterior)
                mostLikelyClass = sorted([i for i in zip(classPosteriors, self.classes)], reverse=True)[0][1]
                predictedClasses.append(mostLikelyClass)

        return predictedClasses