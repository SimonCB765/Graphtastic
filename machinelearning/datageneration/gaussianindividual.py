import argparse
from scipy.stats import multivariate_normal
from scipy import mean

def main(generatingDistributions, outputLocation, numberOfObservations=None, header=True, separator='\t'):
    """Generate a random dataset of a specified number of classes.

    Each feature in the dataset is described by a Gaussian with specified mean and variance of 1.
	The data for each class is generated from a multivariate Gaussian distribution (with covariance matrix = I) defined in this manner.

    :param generatingDistributions:     The mean vectors of the distributions to generate the data from. Each element of the outer list specifies
                                        the mean vector for one class for which data should be generated.
    :type generatingDistributions:      list of lists of floats
    :param outputLocation:              The location where the dataset will be written out.
    :type outputLocation:               string
    :param numberOfObservations:        The number of observations to generate for each class.
    :type numberOfObservations:         list of floats
    :param header:                      Whether a header should be placed on the dataset file.
    :type header:                       boolean
    :param separator:                   The string used to separate values in the dataset file.
    :type separator:                    string

    """

    numberOfClasses = len(generatingDistributions)  # The number of classes for which data should be generated.
    numberOfFeatures = len(generatingDistributions[0])  # The number of features in the dataset to be generated.

    # Setup the number of observations for each class.
    if not numberOfObservations:
        # If the number of observations was not specified for any class, then default to 50 observations per class.
        numberOfObservations = [50 for i in range(numberOfClasses)]
    elif len(numberOfObservations) < numberOfClasses:
        # If the number of observations was specified for at least one class, then default the number of observations for the classes where the number
        # was not specified to the mean of the number of observations for the classes with a specified number.
        meanNumberObservations = int(mean(numberOfObservations))
        numberOfObservations += [meanNumberObservations for i in range(numberOfClasses - len(numberOfObservations))]

    # Generate the data.
    with open(outputLocation, 'w') as writeData:
        if header:
            header = separator.join([str(i) for i in range(numberOfFeatures)] + ['Class'])
            writeData.write(header + '\n')
        for i, j in enumerate(generatingDistributions):
            generator = multivariate_normal(mean=j)  # Define the generating distribution for the class.
            observationsToGenerate = numberOfObservations[i]
            observations = generator.rvs(observationsToGenerate)  # Generate the observations for class i.
            for o in observations:
                writeData.write(separator.join([str(k) for k in o] + [str(i)]) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=('Generates a random dataset containg a specified number of classes. ' +
                                                  'Each feature in the dataset is described by a Gaussian with specified mean and variance of 1. ' +
												  'The data for each class is generated from a multivariate Gaussian distribution (with covariance ' +
												  'matrix = I) defined in this manner.')
									)
    parser.add_argument('means', help='The mean vectors of the generating distributions. Each mean vector must contai the same number of components.')
    parser.add_argument('dataLocation', help='The location where the dataset will be written out.')
    parser.add_argument('-n', '--numObs', help='The number of observations to generate for each class. (Required type: list of integers).',
                        default=None, required=False)
    parser.add_argument('-t', '--header', help='Whether a header should be output on the dataset file. (Default value: No header).',
                        action='store_true', default=False, required=False)
    parser.add_argument('-s', '--sep', help='The separator used when writing out the dataset. (Required type: %(type)s, default value: %(default)s).',
                        type=str, default='\t', required=False)
    args = parser.parse_args()

    numberOfObservations = args.numObs if not args.numObs else eval(args.numObs)
    main(eval(args.means), args.dataLocation, numberOfObservations, header=args.header, separator=args.sep)