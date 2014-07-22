import argparse
import matplotlib.pyplot as plt
import pandas
import sys

import colors


def main(datasetLocation, outputLocation, labelsColumn=None, separator='\t', rowsToPlot=0, coloredLines=False, title='', xLabel='', yLabel=''):
    """Create a scatter plot of a given dataset.

    :param datasetLocation:     The location of the dataset to generate a scatterplot from.
    :type datasetLocation:      str
    :param outputLocation:      The location where the figure should be saved.
    :type outputLocation:       str
    :param separator:           The string that separates values in the file containing dataset.
    :type separator:            str
    :param labelsColumn:        The index of the column containing the labels of the lines (can use negative indexing).
    :type labelsColumn:         int (or None if there is no labels column)
    :param rowsToPlot:          The indices of the rows to plot (defaults to the first row (0th index)).
    :type rowsToPlot:           list of ints
    :param title:               The title for the figure.
    :type title:                str

    """

    # Extract the data.
    dataset = pandas.read_csv(datasetLocation, sep=separator, header=None)

    # Process the data.
    if not labelsColumn == None:
        labels = [i for i in dataset.iloc[:, labelsColumn]] # Extract the labels.
        dataset = dataset.drop(dataset.columns[labelsColumn], axis=1)  # Create a new dataset with the label column dropped.
    else:
        labels = None
    rowData = [dataset.iloc[i, :] for i in rowsToPlot]
    notNullData = [pandas.notnull(i) for i in rowData]
    lineData = [[j[0] for j in zip(i[0], i[1]) if j[1]] for i in zip(rowData, notNullData)]  # A list of lists of x,y pairs. One internal list for each line.
    xData = [[float(j.split(',')[0]) for j in i] for i in lineData]
    yData = [[float(j.split(',')[1]) for j in i] for i in lineData]

    # Plot the data.
    if not labels and not coloredLines:
        lineColorSet = None
    else:
        lineColorSet = 'set2'
    plot(xData, yData, outputLocation, labels=labels, title=title, xLabel=xLabel, yLabel=yLabel, lineColorSet=lineColorSet)


def plot(xValues, yValues, outputLocation=None, labels=None, currentFigure=None, title='', xLabel='', yLabel='', linestyle='-', linewidth=4,
         color='black', marker='o', markersize=40, markeredgewidth=0.25, lineColorSet='set2', colorMapping=None, alpha=0.75,
         spinesToRemove=['top', 'right'], legend=True):
    """Plot a line graph.

    :param xValues:             The x values of the points to plot.
    :type xValues:              list of lists, with each internal list containing the x points for one line (must be same dimensions as yValues)
    :param yValues:             The y values of the points to plot.
    :type yValues:              list of lists, with each internal list containing the y points for one line (must be same dimensions as xValues)
    :param outputLocation:      The location where the figure will be saved.
    :type outputLocation:       str (or None if saving is not desired)
    :param labels:              The label for each line. One label per line with labels[i] being the label for the line with x and y points at xValues[i] and
                                yValues[i] respectively.
    :type labels:               list of strings
    :param currentFigure:       The figure from which the axes to plot the scatterplot on will be taken. If not provided, then a new figure will be created.
    :type currentFigure:        matplotlib.figure.Figure
    :param title:               The title for the plot.
    :type title:                str
    :param xLabel:              The label for the x axis.
    :type xLabel:               str
    :param yLabel:              The label for the y axis.
    :type yLabel:               str
    :param linestyle:           The style of the lines to plot.
    :type linestyle:            any valid style accepted by matplotlib.pyplot.plot
    :param linewidth:           The width of the lines to plot.
    :type linewidth:            float
    :param color:               The color of the lines to plot (only used if there are no labels and lineColorSet == None).
    :type color:                any valid color accepted by matplotlib.pyplot.plot
    :param marker:              The style of marker to place at each join in the lines.
    :type marker:               any valid shape accepted by matplotlib.pyplot.scatter
    :param markersize:          The size of each marker.
    :type markersize:           float
    :param markeredgewidth:     The width of the line edges around the points.
    :type markeredgewidth:      float
    :param lineColorSet:        The color set to use in plotting the points. If colorMapping is provided this parameter is ignored. Otherwise, the
                                color set will be cycled through to assign colors to class values (so should have at least as many colors as there
                                are distinct values).
    :type lineColorSet:         any key in the colors.colorMaps dictionary
    :param colorMapping:        A mapping from class values to their RGB color value.
    :type colorMapping:         dict
    :param alpha:               The alpha value for the points face colors.
    :type alpha:                float between 0 and 1
    :param spinesToRemove:      The spines that should be removed from the axes.
    :type spinesToRemove:       list containing any of ['left', 'right', 'top', 'bottom']
    :param legend:              Whether a legend should be added.
    :type legend:               boolean
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
    plt.xlabel(xLabel, fontsize=16, color='0.25')
    plt.ylabel(yLabel, fontsize=16, color='0.25')

    # Generate the plot.
    if not labels or not lineColorSet:
        for i in zip(xValues, yValues):
            plt.plot(i[0], i[1], linestyle=linestyle, linewidth=linewidth, color=color, marker=marker, markersize=markersize, markeredgewidth=markeredgewidth, alpha=alpha)
    else:
        # Map the lines to colors. If there are more lines than colors in the color set, then multiple lines will be mapped to the same color.
        orderedLabels = sorted(labels)
        if not colorMapping:
            colorsToUse = colors.colorMaps[lineColorSet]
            numberOfColors = len(colorsToUse)
            colorMapping = {}
            for i, j in enumerate(orderedLabels):
                colorMapping[j] = colorsToUse[i % numberOfColors]

        for i in zip(xValues, yValues, labels):
            if marker:
                # Using two scatterplots means that the lines will not intersect with the marker points. Instead there will be a nice white space around
                # each marker with the marker inside it.
                axes.scatter(i[0], i[1], s=markersize*4, c='white', marker='o', edgecolor='none', zorder=1)
                axes.scatter(i[0], i[1], s=markersize, c=colorMapping[i[2]], marker=marker, edgecolor='black', linewidths=markeredgewidth, zorder=2)
            axes.plot(i[0], i[1], label=i[2], linestyle=linestyle, linewidth=linewidth, color=colorMapping[i[2]], alpha=alpha, solid_joinstyle='bevel', zorder=0)

        # Add a legend.
        if legend:
            legend = axes.legend(bbox_to_anchor=(1.05, 0.5), loc=6, borderaxespad=0, frameon=True, scatterpoints=1)
            legendFrame = legend.get_frame()
            legendFrame.set_facecolor('white')
            legendFrame.set_edgecolor('black')
            legendFrame.set_linewidth(0.2)
            for i in legend.get_texts():
                i.set_color('0.25')

    if outputLocation:
        plt.savefig(outputLocation, bbox_inches='tight', transparent=True)
    else:
        return currentFigure, axes


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=('Generate a line graph from a file of a dataset.'),
                                     epilog=('The dataset should contain n rows, one for each line (not all lines need be plotted). Each entry on a ' +
                                             'row should consist of two floats separated by a comma, with the x value first. For example, entry 1.1,2.0 ' +
                                             'will cause the line to pass through point (1.1, 2.0). The rows do not need to have the same number of entries.'))
    parser.add_argument('dataset', help='The location of the dataset file.')
    parser.add_argument('output', help='The location where the image of the plot will be saved.')
    parser.add_argument('-s', '--sep', help='The separator used by the dataset file. (Required type: %(type)s, default value: %(default)s).',
                        type=str, default='\t', required=False)
    parser.add_argument('-l', '--label', help='The index of the column in which the labels for each line are recorded. Only use if different lines should be highlighted on the plot. (Required type: %(type)s, default value: no labels used).',
                        type=int, default=None, required=False)
    parser.add_argument('-r', '--rows', help='The indices of the rows that should be plotted. (Required type: ints separated by commas, default value: first row).',
                        type=str, default='0', required=False)
    parser.add_argument('-c', '--color', help='Whether the lines should be colored (only applies when no labels are given). (Default value: No color).',
                        action='store_true', default=False, required=False)
    parser.add_argument('-t', '--title', help='The title for the plot. (Required type: %(type)s, default value: %(default)s).',
                        type=str, default='', required=False)
    parser.add_argument('-x', '--xLabel', help='The label for the x axis of the plot. (Required type: %(type)s, default value: %(default)s).',
                        type=str, default='', required=False)
    parser.add_argument('-y', '--yLabel', help='The label for the y axis of the plot. (Required type: %(type)s, default value: %(default)s).',
                        type=str, default='', required=False)
    args = parser.parse_args()

    rows = args.rows.split(',')
    if len(rows) < 1:
        print('ERROR: Not enough rows were specified using the -r or --rows flags.')
        sys.exit()
    try:
        rowsToPlot = [int(i) for i in rows]
    except ValueError:
        print('ERROR: Non-integer row index provided. Only integer row indices may be supplied using the -r or --rows flags.')
        sys.exit()

    main(args.dataset, args.output, labelsColumn=args.label, separator=args.sep, rowsToPlot=rowsToPlot, coloredLines=args.color, title=args.title,
         xLabel=args.xLabel, yLabel=args.yLabel)