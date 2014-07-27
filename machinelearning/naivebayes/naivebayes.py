import numpy as np
import os
import pandas


class NaiveBayes:
    """A naive Bayes classifier.

    Classes of observations are always assumed to be in the final column of the dataset.

    """

    def __init__(self, dataset='', headerPresent=False, separator='\t'):
        """Initialise a naive Bayes classifier.

        :param dataset:         The location of the dataset used to train the classifier. The class values are assumed to be in the final column.
        :type dataset:          str
        :param headerPresent:   Whether a single line header is present in the dataset file.
        :type headerPresent:    boolean
        :param separator:       The string that separates values in the file containing dataset.
        :type separator:        str

        """

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
        """Predict the class of a dataset of observations.

        :param dataset:                 The location of the dataset containing the observations that are to have their class values predicted.
        :type dataset:                  pandas.DataFrame() object
        :parm likelihoodDistribution:   The assumed distribution of the values of a feature given the class, e.g. P(X = v | Y = c).
        :type likelihoodDistribution:   str (must be equal to 'Gaussian' for now)
        :return :                       The predicted classes of the observations. The predicted classes are sorted in the same order as the observations.
        :type :                         list

        """

        predictedClasses = []  # Predicted class values.

        if likelihoodDistribution == 'Gaussian':
            # Generate maximum likelihood estimates for the mean and variance of each feature.
            self.parameters = self.groupedByClasses.aggregate([np.mean, np.var])[self.features]
            classMeanVectors = {}
            classVarianceVectors = {}
            for i in self.classes:
                classParameters = self.parameters.xs(i)
                classMeanVector = classParameters.xs('mean', level=1)
                classMeanVectors[i] = classMeanVector
                classVarianceVector = classParameters.xs('var', level=1)
                classVarianceVectors[i] = classVarianceVector

            # Predict the class of each observation.
            for index, series in list(dataset.iterrows()):
                classPosteriors = []
                for j in self.classes:
                    classPrior = self.classPriors.xs(j)
                    classMean = classMeanVectors[j]
                    classVariance = classVarianceVectors[j]
                    classLikelihood = (1 / np.sqrt(2 * np.pi * classVariance)) * np.exp(- np.square(series - classMean) / (2 * classVariance))
                    posterior = classPrior * classLikelihood.prod()
                    classPosteriors.append(posterior)
                mostLikelyClass = sorted([i for i in zip(classPosteriors, self.classes)], reverse=True)[0][1]  # Predicted class is the one with the largest posterior.
                predictedClasses.append(mostLikelyClass)

        return predictedClasses