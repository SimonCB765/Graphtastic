import argparse
import matplotlib.pyplot as plt
from multiprocessing import Pool
import numpy as np
import pandas

import nearestneighbours

# Import the scatterplot script, colors and the script for generating the color fill.
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, parent_dir)
import colors
import discretecolormesh
import scatter


def main(datasetLocation, neighbours, outputLocation, headerPresent=False, separator='\t', classColumn=-1, columnsToPlot=None, title='', divisions=100):
    """Create a scatter plot along with the NN decision boundaries induced by the data.

    :param datasetLocation:     The location of the dataset to generate the plot from.
    :type datasetLocation:      str
    :param neighbours:          The number of neighbours to use in the classification.
    :type neighbours:           int
    :param outputLocation:      The location where the figure should be saved.
    :type outputLocation:       str
    :param headerPresent:       Whether a single line header is present in the dataset file.
    :type headerPresent:        boolean
    :param separator:           The string that separates values in the file containing dataset.
    :type separator:            str
    :param classColumn:         The index of the column containing the class of the observations (can use negative indexing).
    :type classColumn:          int
    :param columnsToPlot:       The indices of the two columns to plot (defaults to the first and second columns [0, 1]).
    :type columnsToPlot:        list of ints
    :param title:               The title for the figure.
    :type title:                str
    :param divisions:           The number of slices to divide each axis into. Each of the divisions^2 (x,y) pairs will be evaluated for its class.
    :type divisions:            int

    """

    # Setup the columns to plot.
    if not columnsToPlot or len(columnsToPlot) != 2:
        # If two columns were not specified, then default to plotting the first two columns.
        columnsToPlot = [0, 1]

    # Extract the desired data.
    dataset = pandas.read_csv(datasetLocation, sep=separator, header=(0 if headerPresent else None))
    dataset = dataset.iloc[:, columnsToPlot + [-1]]

    # Define the color set used.
    colorSet = 'set2'

    # Extract the data to plot.
    featureOne = dataset.iloc[:, columnsToPlot[0]]
    featureTwo = dataset.iloc[:, columnsToPlot[1]]
    classes = dataset.iloc[:, -1]

    # Get axes limits.
    featureOneMin = featureOne.nsmallest(1).iloc[0]
    featureOneMax = featureOne.nlargest(1).iloc[0]
    featureOneRange = (featureOneMax - featureOneMin)
    featureOneMin -= featureOneRange * 0.1
    featureOneMax += featureOneRange * 0.1
    featureTwoMin = featureTwo.nsmallest(1).iloc[0]
    featureTwoMax = featureTwo.nlargest(1).iloc[0]
    featureTwoRange = (featureTwoMax - featureTwoMin)
    featureTwoMin -= featureTwoRange * 0.1
    featureTwoMax += featureTwoRange * 0.1

    # Create the (x,y) pair mesh for the decision boundary and color filling.
    featureOneDelta = (featureOneMax - featureOneMin) / divisions
    featureTwoDelta = (featureTwoMax - featureTwoMin) / divisions
    featureOneSteps = np.arange(featureOneMin, featureOneMax + featureOneDelta, featureOneDelta)
    featureTwoSteps = np.arange(featureTwoMin, featureTwoMax + featureTwoDelta, featureTwoDelta)
    featureOneMesh, featureTwoMesh = np.meshgrid(featureOneSteps, featureTwoSteps)

    # Create the NN classifier.
    classifier = nearestneighbours.NearestNeighbours(dataset, headerPresent=headerPresent)

    # Classify the points on the mesh.
    workerPool = Pool(5)
    classificatons = workerPool.map(worker, [(classifier, pandas.DataFrame([featureOneMesh[:, i], featureTwoMesh[:, i]]).T, neighbours) for i in range(len(featureOneSteps))])

    # Draw the boundary and classification regions. Ideally pcolormesh would be used, but with alpha values this gives unsightly lines along the edges of the
    # mesh due to overlapping squares (http://matplotlib.1069221.n5.nabble.com/Quadmesh-with-alpha-without-the-nasty-edge-effects-td41039.html) that
    # edgecolor='none' does not fix. Instead use my workaround that plays nice with alpha values and gives boundary lines.
    uniqueClasses = sorted(classes.unique())
    colorsToUse = colors.colorMaps[colorSet]
    numberOfColors = len(colorsToUse)
    classToColorMapping = {}
    for i, j in enumerate(uniqueClasses):
        classToColorMapping[j] = colorsToUse[i % numberOfColors]
    figure, axes = discretecolormesh.main(featureOneMesh, featureTwoMesh, np.transpose(np.array(classificatons)), border=True, classToColorMapping=classToColorMapping, fillAlpha=0.3, boundaryWidth=1)

    # Plot the data.
    figure, axes = scatter.plot(featureOne, featureTwo, classLabels=classes, currentFigure=figure, title=title, xLabel=dataset.columns[0], yLabel=dataset.columns[1], faceColorSet=colorSet, alpha=1.0)

    # Reset the axes to ensure that the color mesh plotting hasn't changed it, which would cause a bunch of whitespace around the edges of the plot.
    plt.axis('scaled')
    axes.set_xlim([featureOneMin, featureOneMax])
    axes.set_ylim([featureTwoMin, featureTwoMax])

    # Save the figure.
    plt.savefig(outputLocation, bbox_inches='tight', transparent=True)


def worker(parameters):
    """Classify the dataset using the supplied classifier and number of neighbours.

    Auxiliary function is required to meet restrictions placed on Pool.map.

    :param parameters:      The classifier, dataset and number of nearest neighbours to use in the classification.
    :type parameters:       list of a nearestneighbours.NearestNeighbours object, pandas.DataFrame object and an int
    :returns :              The classes of the observations in the input dataset
    :type :                 list

    """

    classifier = parameters[0]
    data = parameters[1]
    k = parameters[2]
    return classifier.classify_data(data, k)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=('Generate a the decison boundary for a k-NN classifier.'))
    parser.add_argument('dataset', help='The location of the dataset file.')
    parser.add_argument('neighbours', help='The number k of neighbours to use. (Required type: %(type)s, default value: %(default)s).',
                        type=int, default=3)
    parser.add_argument('output', help='The location where the image of the plot will be saved.')
    parser.add_argument('-r', '--header', help='Whether a header is present in dataset file. (Default value: No header).',
                        action='store_true', default=False, required=False)
    parser.add_argument('-s', '--sep', help='The separator used by the dataset file. (Required type: %(type)s, default value: %(default)s).',
                        type=str, default='\t', required=False)
    parser.add_argument('-d', '--classCol', help='The index of the column in which the values of the class variable can be found. (Required type: %(type)s, default value: %(default)s).',
                        type=int, default=-1, required=False)
    parser.add_argument('-c', '--cols', help='The indices of the columns that should be plotted against each other. (Required type: two ints separated by a comma, default value: first two columns).',
                        type=str, default='0,1', required=False)
    parser.add_argument('-t', '--title', help='The title for the plot. (Required type: %(type)s, default value: %(default)s).',
                        type=str, default='', required=False)
    parser.add_argument('-x', '--sectors', help='Number of sectors to divide each axis into for forming the classification areas. More sectors means smoother boundaries but longer computation times. (Required type: %(type)s, default value: %(default)s).',
                        type=float, default=100, required=False)
    args = parser.parse_args()

    columnsToPlot = None if len(args.cols.split(',')) != 2 else [int(i) for i in args.cols.split(',')]
    main(args.dataset, args.neighbours, args.output, headerPresent=args.header, separator=args.sep, classColumn=args.classCol, columnsToPlot=columnsToPlot, title=args.title, divisions=args.sectors)