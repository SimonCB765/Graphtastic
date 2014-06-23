import os
import pandas

import metrics


class NearestNeighbours:
    """A nearest neighbours classifier.

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

    def add_and_classify_data(self, dataset, k=3, metric='Euclidean'):
        """Classify a dataset using the stored dataset, and then add the new dataset to the stored one.

        :param dataset:     The dataset of observations to classify. Assumes that features are the columns and observations the rows.
        :type dataset:      An object that can be converted into a pandas.DataFrame.
        :param k:           The number of neighbours to use in the classification.
        :type k:            int
        :param metric:      The distance metric to use.
        :type metric:       string
        :returns :          The classification of each observation. The classification are in the same order as the observations in the dataset to classify.
        :type :             list

        """

        classifications = self.classify_data(dataset, metric, k)
        self.add_data(dataset, classifications)
        return classifications

    def add_data(self, dataset, classifications=None):
        """Add new data to the stored dataset.

        :param dataset:             The dataset of observations to add.
        :type dataset:              An object that can be converted into a pandas.DataFrame. If you want to add a single observation that is not in the
                                    format of a pandas.DataFrame, then it needs to able to be turned into a row (rather than a column) when
                                    pandas.DataFrame(dataset) is called (e.g. [1, 2, 3] becomes a column vector but [[1, 2, 3]] becomes a row vector).
        :param classifications:     The classes of the added datapoints (if they are not already present in the input dataset).
        :type classifications:      If None, then classifications must be provided as the last column of dataset. Otherwise, it contains an object that
                                    will be converted to a column vector when pandas.DataFrame(classifications) is called.

        """


        if classifications:
            dataset = pandas.DataFrame(dataset, columns=self.dataset.columns[:-1])
            classCol = pandas.DataFrame(classifications, columns=['Class'])
            dataset = pandas.concat([dataset, classCol], axis=1)
        else:
            dataset = pandas.DataFrame(dataset, columns=self.dataset.columns)
        self.dataset = pandas.concat([self.dataset, dataset], axis=0)

    def classify_data(self, dataset, k=3, metric='Euclidean'):
        """Classify a dataset using the stored dataset.

        :param dataset:     The dataset of observations to classify. Assumes that features are the columns and observations the rows.
        :type dataset:      An object that can be converted into a pandas.DataFrame.
        :param k:           The number of neighbours to use in the classification.
        :type k:            int
        :param metric:      The distance metric to use.
        :type metric:       string
        :returns :          The classification of each observation. The classification are in the same order as the observations in the dataset to classify.
        :type :             list

        """

        dataset = pandas.DataFrame(dataset)
        distanceMetric = metrics.metrics[metric]
        classifications = []
        classes = self.dataset.iloc[:, -1]  # Extract the class of each stored observation.
        for index, series in dataset.iterrows():
            distances = self.dataset.apply(lambda x : distanceMetric(x[:-1], series), axis=1)  # Determine each input observation's distance from all stored observations.
            classification = classes.loc[distances.nsmallest(k).index].value_counts().index[0]  # Ties in the number of nearest neighbours are broken arbitrarily.
            classifications.append(classification)
        return classifications

    def get_neighbours(self, observation, k=None, metric='Euclidean'):
        """Get the distance from a given observation to a set of the observations in the stored dataset.

        :param observation:     The observation for which all distances to stored observations will be calculated.
        :type observation:      1 dimensional array like object with features ordered the same as self.dataset
        :param k:               The number of neighbours to return distances for. If None, then return distances for all observations in the stored dataset.
        :type k:                int or None
        :param metric:          The distance metric to use.
        :type metric:           string
        :return :               The desired number of stored observations, along with their distance to the input observations.
        :type :                 pandas.DataFrame

        """

        distanceMetric = metrics.metrics[metric]
        distances = self.dataset.apply(lambda x : distanceMetric(x[:-1], observation), axis=1)  # Determine each stored observation's distance from the input observation.
        distances = pandas.Series(distances)
        if k:
            distanceToNearest = distances.nsmallest(k)
            nearestNeighours = self.dataset.loc[distanceToNearest.index]
            returnValue = pandas.concat([nearestNeighours, distanceToNearest], axis=1)
        else:
            returnValue =  pandas.concat([self.dataset, distances], axis=1)
        # Change columns in a depressingly involved manner.
        newColumns = returnValue.columns.values
        newColumns[-1] = 'Distance'
        returnValue.columns  = newColumns
        return returnValue