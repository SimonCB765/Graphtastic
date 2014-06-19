import argparse
import matplotlib.colors as col
import matplotlib.pyplot as plt
from multiprocessing import Pool
import numpy as np
import pandas

import nearestneighbours

# Import the scatterplot script.
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, parent_dir)
import scatter
import colors


def main(datasetLocation, neighbours, outputLocation, headerPresent=False, separator='\t', columnsToPlot=None, title=''):
    """
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

    # Extract the data to plot and plot it.
    featureOne = dataset.iloc[:, 0]
    featureTwo = dataset.iloc[:, 1]
    classes = dataset.iloc[:, -1]
    figure, axes = scatter.plot(featureOne, featureTwo, classLabels=classes, title=title, xLabel=dataset.columns[0], yLabel=dataset.columns[1], faceColorSet=colorSet)

    # Get axis limits.
    featureOneMin = axes.get_xlim()[0]
    featureOneMax = axes.get_xlim()[1]
    featureTwoMin = axes.get_ylim()[0]
    featureTwoMax = axes.get_ylim()[1]

    # Create the mesh for the decision boundary.
    delta = 0.05
    featureOneRange = np.arange(featureOneMin, featureOneMax + delta, delta)
    featureTwoRange = np.arange(featureTwoMin, featureTwoMax + delta, delta)
    featureOneMesh, featureTwoMesh = np.meshgrid(featureOneRange, featureTwoRange)

    # Create the NN classifier.
    classifier = nearestneighbours.NearestNeighbours(dataset, headerPresent=headerPresent)

    # Classify the points on the mesh.
    p = Pool(5)
    h = p.map(worker, [(classifier, pandas.DataFrame([featureOneMesh[:, i], featureTwoMesh[:, i]]).T, neighbours) for i in range(len(featureOneRange))])

    # Draw the boundary and classification regions.
    uniqueClasses = sorted(classes.unique())
    numberOfClasses = len(uniqueClasses)
    m = plt.pcolormesh(featureOneMesh, featureTwoMesh, np.transpose(np.array(h)), cmap=col.ListedColormap(colors.colorMaps[colorSet][:numberOfClasses]), edgecolors='None', alpha=0.3, zorder=-1)

    # Save the figure.
    plt.savefig(outputLocation, bbox_inches='tight', transparent=True)


def worker(parameters):
    """
    So complicated to get around restrictions on Pool.map
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
    parser.add_argument('-c', '--cols', help='The indices of the columns that should be plotted against each other. (Required type: two ints separated by a comma, default value: first two columns).',
                        type=str, default='0,1', required=False)
    parser.add_argument('-t', '--title', help='The title for the plot. (Required type: %(type)s, default value: %(default)s).',
                        type=str, default='', required=False)
    args = parser.parse_args()

    columnsToPlot = None if len(args.cols.split(',')) != 2 else [int(i) for i in args.cols.split(',')]
    main(args.dataset, args.neighbours, args.output, headerPresent=args.header, separator=args.sep, columnsToPlot=columnsToPlot, title=args.title)