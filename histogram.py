import argparse
import matplotlib.patches as patches
import matplotlib.path as path
import matplotlib.pyplot as plt
import pandas
import sys

import colors


def main(datasetLocation, outputLocation, headerPresent=False, separator='\t', title='', direction='Up', columnToPlot=0, bins=10):
    """Create a scatter plot of a given dataset.

    :param datasetLocation:     The location of the dataset to generate a scatterplot from.
    :type datasetLocation:      str
    :param outputLocation:      The location where the figure should be saved.
    :type outputLocation:       str
    :param headerPresent:       Whether a single line header is present in the dataset file.
    :type headerPresent:        boolean
    :param separator:           The string that separates values in the file containing dataset.
    :type separator:            str
    :param title:               The title for the figure.
    :type title:                str
    :param direction:           The direction that the histogram should be plotted. One of 'Up', 'Down', 'Left' or 'Rght'.
    :type direction:            str
    :param columnToPlot:        The column containing the data for the histogram.
    :type columnToPlot:         int
    :param bins:                The number of equally spaced bins to use.
    :type bins:                 int

    """

    # Extract the data.
    dataset = pandas.read_csv(datasetLocation, sep=separator, header=(0 if headerPresent else None))

    # Extract the data to plot and plot it.
    data = dataset.iloc[:, columnToPlot]
    plot(data, bins=bins, outputLocation=outputLocation, title=title, xLabel=dataset.columns[columnToPlot], yLabel='Counts')


def plot(data, bins=10, direction='Up', outputLocation=None, currentFigure=None, title='', xLabel='', yLabel='', edgeColor='none',
         faceColor='black', linewidth=1, alpha=0.5, spinesToRemove=['top', 'right']):
    """Generate a histogram.

    When generating N bins, the first n - 1 bins will contain values >= their left edge and < their right edge. The final rightmost bin will contain all
    the remaining values >= its left edge (i.e. its right edge is included in the range of values it holds unlike the other bins).

    :param data:                The data to be plotted.
    :type data:                 pandas.DataFrame() column vector
    :param bins:                The number of bins to put the data in.
    :type bins:                 int
    :param direction:           The direction that the bars should go.
    :type direction:            one of 'Up', 'Down', 'Left' or 'Right'
    :param outputLocation:      The location where the figure will be saved.
    :type outputLocation:       str (or None if saving is not desired)
    :param currentFigure:       The figure from which the axes to plot the scatterplot on will be taken. If not provided, then a new figure will be created.
    :type currentFigure:        matplotlib.figure.Figure
    :param title:               The title for the plot.
    :type title:                str
    :param xLabel:              The label for the x axis.
    :type xLabel:               str
    :param yLabel:              The label for the y axis.
    :type yLabel:               str
    :param edgeColor:           The color of the line edges around the bars.
    :type edgeColor:            any color accepted by the matplotlib.patches.PathPatch edgecolor parameter
    :param faceColor:           The color for the face of the bars.
    :type faceColor:            any color accepted by the matplotlib.patches.PathPatch facecolor parameter
    :param linewidth:           The width of the line edges around the bars.
    :type linewidth:            float
    :param alpha:               The alpha value for the face color of the bars.
    :type alpha:                float between 0 and 1
    :param spinesToRemove:      The spines that should be removed from the axes.
    :type spinesToRemove:       list containing any of ['left', 'right', 'top', 'bottom']
    :returns :                  The figure and axes on which the scatterplot was plotted if saving is not to be performed.
    :type :                     objects of type matplotlib.figure.Figure, matplotlib.axes.Axes

    """

    # Get the axes the will be used for the plot.
    try:
        axes = currentFigure.gca()
    except AttributeError:
        # If the figure is not given then create it.
        currentFigure = plt.figure()
        axes = currentFigure.add_subplot(1, 1, 1)

    # Remove desired spines.
    for i in spinesToRemove:
        axes.spines[i].set_visible(False)

    # Change remaining spines' widths and colors.
    for i in set(['left', 'right', 'top', 'bottom']) - set(spinesToRemove):
        axes.spines[i].set_linewidth(0.75)
        axes.spines[i].set_color('0.25')

    # Remove ticks from the axes and soften the color of the labels slightly.
    axes.xaxis.set_ticks_position('none')
    for i in axes.xaxis.get_ticklabels():
        i.set_color('0.25')
    axes.yaxis.set_ticks_position('none')
    for i in axes.yaxis.get_ticklabels():
        i.set_color('0.25')

    # Create the figure title.
    axes.set_title(title, fontsize=22, color='0.25')

    # Label the axes.
    if direction in ['Up', 'Down']:
        plt.xlabel(xLabel, fontsize=16, color='0.25')
        plt.ylabel(yLabel, fontsize=16, color='0.25')
    else:  #if direction in ['Left', 'Right']:
        plt.xlabel(yLabel, fontsize=16, color='0.25')
        plt.ylabel(xLabel, fontsize=16, color='0.25')

    # Determine the bin width.
    minValue = data.nsmallest(1).iloc[0]
    maxValue = data.nlargest(1).iloc[0]
    binWidth = (maxValue - minValue) / bins

    # Bin the data.
    leftBinEdges = [minValue + (binWidth * i) for i in range(bins)]
    countEdges = leftBinEdges + [maxValue + 1]
    binCounts = [(data[data >= countEdges[i]] < countEdges[i + 1]).sum() for i in range(len(leftBinEdges))]

    # Determine the vertices for the histogram.
    extendedEdges = leftBinEdges + [maxValue]
    leftEdges = [i + (binWidth * 0.05) for i in leftBinEdges]
    rightEdges = [i - (binWidth * 0.05) for i in extendedEdges[1:]]
    bottomLeftEdges = [[i, 0] for i in leftEdges]
    topLeftEdges = [[i[0], i[1]] for i in zip(leftEdges, binCounts)]
    topRightEdges = [[i[0], i[1]] for i in zip(rightEdges, binCounts)]
    bottomRightEdges = [[i, 0] for i in rightEdges]
    vertices = [None for i in range(bins * 5)]  # 5 vertices per bar: 1 for the MOVETO, 3 for the LINETO, and 1 for the CLOSEPOLY.
    vertices[0::5] = bottomLeftEdges
    vertices[1::5] = topLeftEdges
    vertices[2::5] = topRightEdges
    vertices[3::5] = bottomRightEdges
    vertices[4::5] = bottomLeftEdges

    # Transform the vertices to deal with plotting from the left or right.
    if direction in ['Left', 'Right']:
        vertices = [[i[1], i[0]] for i in vertices]

    # Create the path codes.
    codes = [path.Path.LINETO for i in range(bins * 5)]  # 5 codes per bar: 1 for the MOVETO, 3 for the LINETO, and 1 for the CLOSEPOLY.
    codes[0::5] = [path.Path.MOVETO for i in binCounts]
    codes[4::5] = [path.Path.CLOSEPOLY for i in binCounts]

    # Plot the histogram.
    histoPath = path.Path(vertices, codes)
    histoPatch = patches.PathPatch(histoPath, facecolor=faceColor, edgecolor=edgeColor, linewidth=linewidth, alpha=alpha)
    axes.add_patch(histoPatch)

    # Transform the axes to account for the direction desired, and scale the axes if needed.
    if direction == 'Up':
        # Only scaling needed.
        scale_axes(axes, xMin=(minValue - binWidth), xMax=(maxValue + binWidth), yMin=0, yMax=max(binCounts) + (0.1 * max(binCounts)))
    elif direction == 'Left':
        scale_axes(axes, xMin=0, xMax=max(binCounts) + (0.1 * max(binCounts)), yMin=(minValue - binWidth), yMax=(maxValue + binWidth))
        axes.invert_xaxis()
    elif direction == 'Down':
        scale_axes(axes, xMin=(minValue - binWidth), xMax=(maxValue + binWidth), yMin=0, yMax=max(binCounts) + (0.1 * max(binCounts)))
        axes.invert_yaxis()
    else:  #if direction == 'Right':
        scale_axes(axes, xMin=0, xMax=max(binCounts) + (0.1 * max(binCounts)), yMin=(minValue - binWidth), yMax=(maxValue + binWidth))

    if outputLocation:
        plt.savefig(outputLocation, bbox_inches='tight', transparent=True)
    else:
        return currentFigure, axes


def scale_axes(axes, xMin, xMax, yMin, yMax):
    """Scale the axes.

    :param axes:    The axes instance that is to be scaled.
    :type axes:     an object of type matplotlib.axes
    :param xMin:    The minimum value for the x axis.
    :type xMin:     float
    :param xMax:    The maximum value for the x axis.
    :type xMax:     float
    :param yMin:    The minimum value for the y axis.
    :type yMin:     float
    :param yMax:    The maximum value for the y axis.
    :type yMax:     float

    """

    xLimits = axes.get_xlim()
    newXLimits = [xLimits[0], xLimits[1]]
    if xLimits[0] > xMin:
        newXLimits[0] = xMin
    if xLimits[1] < xMax:
        newXLimits[1] = xMax
    axes.set_xlim(newXLimits)
    yLimits = axes.get_ylim()
    newYLimits = [yLimits[0], yLimits[1]]
    if yLimits[0] > yMin:
        newYLimits[0] = yMin
    if yLimits[1] < yMax:
        newYLimits[1] = yMax
    axes.set_ylim(newYLimits)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=('Generate a histogram from a file of a dataset.'),
                                     epilog=('The dataset should be saved as an n x p matrix, where there are n rows of observations and p columns ' +
                                            'of variables.'))
    parser.add_argument('dataset', help='The location of the dataset file.')
    parser.add_argument('output', help='The location where the image of the plot will be saved.')
    parser.add_argument('-r', '--header', help='Whether a header is present in the dataset file. (Default value: No header).',
                        action='store_true', default=False, required=False)
    parser.add_argument('-s', '--sep', help='The separator used by the dataset file. (Required type: %(type)s, default value: %(default)s).',
                        type=str, default='\t', required=False)
    parser.add_argument('-t', '--title', help='The title for the plot. (Required type: %(type)s, default value: %(default)s).',
                        type=str, default='', required=False)
    parser.add_argument('-d', '--dir', help='The direction for the histogram. (Required type: %(type)s, default value: %(default)s).',
                        type=str, default='Up', choices=['Up', 'Down', 'Left', 'Right'], required=False)
    parser.add_argument('-c', '--col', help='The index of the column containing the data for the histogram (negative indexing permitted). (Required type: %(type)s, default value: %(default)s).',
                        type=int, default=0, required=False)
    parser.add_argument('-b', '--bins', help='The number of equally spaced bins to use. (Required type: %(type)s, default value: %(default)s).',
                        type=int, default=10, required=False)
    args = parser.parse_args()

    main(args.dataset, args.output, headerPresent=args.header, separator=args.sep, title=args.title, direction=args.dir, columnToPlot=args.col,
         bins=args.bins)