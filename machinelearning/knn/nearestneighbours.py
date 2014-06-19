import os
import pandas

import metrics

class NearestNeighbours:
    """A nearest neighbours classifier.

    Assumes that the class in the datasets used for classification are in the last column.

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
        :param classifications:     The classes of the added datapoints (if they are not already present in the dataset).
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

        :param dataset:     The dataset of observations to classify.
        :type dataset:      An object that can be converted into a pandas.DataFrame.

        """

        dataset = pandas.DataFrame(dataset)
        distanceMetric = metrics.metrics[metric]
        classifications = []
        classes = self.dataset.iloc[:, -1]
        for index, series in dataset.iterrows():
            distances = self.dataset.apply(lambda x : distanceMetric(x[:-1], series), axis=1)
            classification = classes.loc[distances.nsmallest(k).index].value_counts().index[0]
            classifications.append(classification)
        return classifications