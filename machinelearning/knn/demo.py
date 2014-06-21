import argparse
#import matplotlib.colors as col
import matplotlib.pyplot as plt
from multiprocessing import Pool
import numpy as np
import pandas

import nearestneighbours

# Import the scatterplot script and colors.
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, parent_dir)
import scatter
import colors
import discretecolormesh


def main(datasetLocation, neighbours, outputLocation, headerPresent=False, separator='\t', columnsToPlot=None, title='', divisions=100):
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

    # Extract the data to plot.
    featureOne = dataset.iloc[:, 0]
    featureTwo = dataset.iloc[:, 1]
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

    # Create the mesh for the decision boundary.
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

    # Draw the boundary and classification regions. Ideally use pcolormesh, but with alpha value this gives unsightly lines along the edges of the mesh
    # due to overlapping squares (http://matplotlib.1069221.n5.nabble.com/Quadmesh-with-alpha-without-the-nasty-edge-effects-td41039.html) that
    # edgecolor='none' does not fix. Instead use my workaround that provides nicer boundaries and gives boundary lines.
    uniqueClasses = sorted(classes.unique())
#    numberOfClasses = len(uniqueClasses)
#    plt.pcolormesh(featureOneMesh, featureTwoMesh, np.transpose(np.array(classificatons)), cmap=col.ListedColormap(colors.colorMaps[colorSet][:numberOfClasses]), edgecolor='none', alpha=0.3, zorder=-1)
    colorsToUse = colors.colorMaps[colorSet]
    numberOfColors = len(colorsToUse)
    classToColorMapping = {}
    for i, j in enumerate(uniqueClasses):
        classToColorMapping[j] = colorsToUse[i % numberOfColors]
    figure, axes = discretecolormesh.main(featureOneMesh, featureTwoMesh, np.transpose(np.array(classificatons)), border=True, classToColorMapping=classToColorMapping, fillAlpha=0.3, boundaryWidth=1)

    # Plot the data.
    figure, axes = scatter.plot(featureOne, featureTwo, classLabels=classes, currentFigure=figure, title=title, xLabel=dataset.columns[0], yLabel=dataset.columns[1], faceColorSet=colorSet, alpha=1.0)

    # Reset the axes to ensure that the color map plotting hasn't changed it, which would cause a bunch of whitespace around the edges of the color map.
    plt.axis('scaled')
    axes.set_xlim([featureOneMin, featureOneMax])
    axes.set_ylim([featureTwoMin, featureTwoMax])

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
    parser.add_argument('-x', '--sectors', help='Number of sectors to divide each axis into for forming the classification areas. More sectors means smoother boundaries but longer computation times. (Required type: %(type)s, default value: %(default)s).',
                        type=float, default=100, required=False)
    args = parser.parse_args()

    columnsToPlot = None if len(args.cols.split(',')) != 2 else [int(i) for i in args.cols.split(',')]
    main(args.dataset, args.neighbours, args.output, headerPresent=args.header, separator=args.sep, columnsToPlot=columnsToPlot, title=args.title, divisions=args.sectors)