import argparse
from random import choice
from scipy.stats import multivariate_normal
from scipy import mean

def main(meanGeneratingDistributions, draws, outputLocation, variance=0.2, numberOfObservations=None, header=True, separator='\t'):
    """Generate a random dataset of a specified number of classes.

    For the ith class the data is generated as follows:
    1) Create a multivariate Gaussian with mean=meanGeneratingDistributions[i] and covariance matrix=I.
    2) From the distribution in 1, draw draws[i] values.
    3) For each value, j, drawn in 2, define a new multivariate Gaussian with mean j and covariance matrix (I * variance). The variance should be
       defined such that the Gaussians in the new distribution have small variance.
    4) For each of the numberOfObservations[i] observations to generate, uniformly choose a distribution generated in 3 and draw a value from it.

    :param meanGeneratingDistributions:     The mean vectors of the distributions to draw the means of the distributions from which the data will be drawn.
                                            Each element of the outer list specifies the mean vector for one class for which data should be generated.
    :type meanGeneratingDistributions:      list of lists of floats
    :param draws:                           The number of means to draw for each class from its mean generating distribution.
    :type draws:                            list of ints
    :param outputLocation:                  The location where the dataset will be written out.
    :type outputLocation:                   string
    :param variance:                        The variance of the individual uncorrelated Gaussians in the data generating distributions.
    :type variance:                         float
    :param numberOfObservations:            The number of observations to generate for each class.
    :type numberOfObservations:             list of floats
    :param header:                          Whether a header should be placed on the dataset file.
    :type header:                           boolean
    :param separator:                       The string used to separate values in the dataset file.
    :type separator:                        string

    """

    numberOfClasses = len(meanGeneratingDistributions)  # The number of classes for which data should be generated.
    numberOfFeatures = len(meanGeneratingDistributions[0])  # The number of features in the dataset to be generated.

    # Setup the number of observations for each class.
    if not numberOfObservations:
        # If the number of observations was not specified for any class, then default to 50 observations per class.
        numberOfObservations = [50 for i in range(numberOfClasses)]
    elif len(numberOfObservations) < numberOfClasses:
        # If the number of observations was specified for at least one class, then default the number of observations for the classes where the number
        # was not specified to the mean of the number of observations for the classes with a specified number.
        meanNumberObservations = int(mean(numberOfObservations))
        numberOfObservations += [meanNumberObservations for i in range(numberOfClasses - len(numberOfObservations))]

    # Setup the number of means that will be drawn for each class.
    if not draws:
        # If the number of draws is an empty list, then default to 2 draws per class (i.e. a mixture of two Gaussians).
        draws = [2 for i in range(numberOfClasses)]
    elif len(draws) < numberOfClasses:
        # If the number of draws was specified for at least one class, then default the number of draws for the classes where the number
        # was not specified to the mean of the number of draws for the classes with a specified number.
        meanNumberDraws = int(mean(draws))
        draws += [meanNumberDraws for i in range(numberOfClasses - len(draws))]

    # Generate the data.
    with open(outputLocation, 'w') as writeData:
        if header:
            header = separator.join([str(i) for i in range(numberOfFeatures)] + ['Class'])
            writeData.write(header + '\n')
        for i, j in enumerate(meanGeneratingDistributions):
            meanGenerator = multivariate_normal(mean=j)  # Define the generating distribution for the means of the Gaussians for the class.
            generators = [multivariate_normal(mean=meanGenerator.rvs(1), cov=variance) for k in range(draws[i])]
            observationsToGenerate = numberOfObservations[i]
            observations = [choice(generators).rvs(1) for i in range(observationsToGenerate)]  # Generate the observations for class i.
            for o in observations:
                writeData.write(separator.join([str(k) for k in o] + [str(i)]) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=('Generates a random dataset containg a specified number of classes. ' +
                                                  'For the ith class the data is generated as follows: ' +
                                                  '1) Create a multivariate Gaussian with mean=means[i] and covariance matrix=I. ' +
                                                  '2) From the distribution in 1, draw draws[i] values. ' +
                                                  '3) For each value, j, drawn in 2, define a new multivariate Gaussian with mean j and covariance matrix ' +
                                                  '(I * variance). The variance should be defined such that the Gaussians in the new distribution have ' +
                                                  'small variance. ' +
                                                  '4) For each of the numObs[i] observations to generate, uniformly choose a distribution ' +
                                                  'generated in 3 and draw a value from it.'))
    parser.add_argument('means', help='The mean vectors of the generating distributions. Each mean vector must contai the same number of components. (Required type: list of list of floats).')
    parser.add_argument('draws', help='The number of means to draw from the generating distributions. (Required type: list of integers, default value for each class: draw 2 means).')
    parser.add_argument('dataLocation', help='The location where the dataset will be written out.')
    parser.add_argument('-n', '--numObs', help='The number of observations to generate for each class. (Required type: comma delimted list, default value for each class: draw 50 observations).',
                        default=None, required=False)
    parser.add_argument('-t', '--header', help='Whether a header should be output on the dataset file. (Default value: No header).',
                        action='store_true', default=False, required=False)
    parser.add_argument('-s', '--sep', help='The separator used when writing out the dataset. (Required type: %(type)s, default value: %(default)s).',
                        type=str, default='\t', required=False)
    parser.add_argument('-v', '--var', help='The variance for the data generating distributions. (Required type: %(type)s, default value: %(default)s).',
                        type=float, default=0.2, required=False)
    args = parser.parse_args()

    numberOfObservations = args.numObs if not args.numObs else [int(i) for i in args.numObs.split(',')]
    main(eval(args.means), eval(args.draws), args.dataLocation, args.var, numberOfObservations, header=args.header, separator=args.sep)