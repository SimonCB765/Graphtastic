import argparse
import matplotlib.pyplot as plt
import pandas

import colors


def main(datasetLocation, outputLocation, headerPresent=False, separator='\t', classColumn=None, columnsToPlot=None, title=''):
    """Create a scatter plot of a given dataset.

    :param datasetLocation:     The location of the dataset to generate a scatterplot from.
    :type datasetLocation:      str
    :param outputLocation:      The location where the figure should be saved.
    :type outputLocation:       str
    :param headerPresent:       Whether a single line header is present in the dataset file.
    :type headerPresent:        boolean
    :param separator:           The string that separates values in the file containing dataset.
    :type separator:            str
    :param classColumn:         The index of the column containing the class of the observations (can use negative indexing).
    :type classColumn:          int (or None if there is no class)
    :param columnsToPlot:       The indices of the two columns to plot (defaults to the first and second columns [0, 1]).
    :type columnsToPlot:        list of ints
    :param title:               The title for the figure.
    :type title:                str

    """

    # Extract the data.
    dataset = pandas.read_csv(datasetLocation, sep=separator, header=(0 if headerPresent else None))

    # Setup the columns to plot.
    if not columnsToPlot or len(columnsToPlot) != 2:
        # If two columns were not specified, then default to plotting the first two columns.
        columnsToPlot = [0, 1]

    # Extract the data to plot and plot it.
    featureOne = dataset.iloc[:, columnsToPlot[0]]
    featureTwo = dataset.iloc[:, columnsToPlot[1]]
    if type(classColumn) == int:
        # If a class columns has been specified then the classes should be highlighted in the plot.
        classes = dataset.iloc[:, classColumn]
        plot(featureOne, featureTwo, outputLocation, classLabels=classes, title=title, xLabel=dataset.columns[columnsToPlot[0]], yLabel=dataset.columns[columnsToPlot[1]])
    else:
        plot(featureOne, featureTwo, outputLocation, title=title, xLabel=dataset.columns[columnsToPlot[0]], yLabel=dataset.columns[columnsToPlot[1]])


def plot(xValues, yValues, outputLocation=None, classLabels=pandas.Series(), currentFigure=None, title='', xLabel='', yLabel='', size=40,
         shape='o', edgeColor='black', faceColorSet='set2', colorMapping=None, linewidths=0.25, alpha=0.75, spinesToRemove=['top', 'right'], legend=True):
    """Plot a scatterplot.

    :param xValues:             The x values of the points to plot.
    :type xValues:              pandas.DataFrame() column vector
    :param yValues:             The y values of the points to plot.
    :type yValues:              pandas.DataFrame() column vector
    :param outputLocation:      The location where the figure will be saved.
    :type outputLocation:       str (or None if saving is not desired)
    :param classLabels:         The classifications of each observations. Must be ordered in the same order as xValues and yValues.
                                An empty Series indicates no class information should be use.
    :type classLabels:          a pandas object of the same size as xValues and yValues on which unique and empty can be called.
    :param currentFigure:       The figure from which the axes to plot the scatterplot on will be taken. If not provided, then a new figure will be created.
    :type currentFigure:        matplotlib.figure.Figure
    :param title:               The title for the plot.
    :type title:                str
    :param xLabel:              The label for the x axis.
    :type xLabel:               str
    :param yLabel:              The label for the y axis.
    :type yLabel:               str
    :param size:                The size of the points in the scatterplot.
    :type size:                 int
    :param shape:               The shape of the points in the scatterplot.
    :type shape:                any valid shape accepted by matplotlib.pyplot.scatter
    :param edgeColor:           The color of the line edges around the points.
    :type edgeColor:            any color accepted by matplotlib.pyplot.scatter
    :param faceColorSet:        The color set to use in plotting the points. If colorMapping is provided this parameter is ignored. Otherwise, the
                                color set will be cycled through to assign colors to class values (so should have at least as many colors as there
                                are distinct values).
    :type faceColorSet:         any key in the colors.colorMaps dictionary
    :param colorMapping:        A mapping from class values to their RGB color value.
    :type colorMapping:         dict
    :param linewidths:          The width of the line edges around the points.
    :type linewidths:           float
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

    # Change remaining spines' widths and color.
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
    if classLabels.empty:
        # If there are no classes, then generate a basic scatterplot where all points are one color.
        axes.scatter(xValues, yValues, s=size, c='red', marker=shape, edgecolor=edgeColor, linewidths=linewidths, alpha=alpha)
    else:
        # Map the class values to colors. If there are more class values than colors, then multiple lass values will be mapped to the same color.
        uniqueLabels = sorted(classLabels.unique())
        if not colorMapping:
            colorsToUse = colors.colorMaps[faceColorSet]
            numberOfColors = len(colorsToUse)
            colorMapping = {}
            for i, j in enumerate(uniqueLabels):
                colorMapping[j] = colorsToUse[i % numberOfColors]

        for i, j in enumerate(uniqueLabels):
            axes.scatter(xValues[classLabels == j], yValues[classLabels == j], s=size, c=colorMapping[j], label=str(j), marker=shape, edgecolor=edgeColor, linewidths=linewidths, alpha=alpha)
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
    parser = argparse.ArgumentParser(description=('Generate a scatterplot from a file of a dataset.'))
    parser.add_argument('dataset', help='The location of the dataset file.')
    parser.add_argument('output', help='The location where the image of the plot will be saved.')
    parser.add_argument('-r', '--header', help='Whether a header is present in dataset file. (Default value: No header).',
                        action='store_true', default=False, required=False)
    parser.add_argument('-s', '--sep', help='The separator used by the dataset file. (Required type: %(type)s, default value: %(default)s).',
                        type=str, default='\t', required=False)
    parser.add_argument('-d', '--classCol', help='The index of the column in which the values of the class variable can be found. Only use if different classes should be highlighted on the plot. (Required type: %(type)s, default value: no classes used).',
                        type=int, default=None, required=False)
    parser.add_argument('-c', '--cols', help='The indices of the columns that should be plotted against each other. (Required type: two ints separated by a comma, default value: first two columns).',
                        type=str, default='0,1', required=False)
    parser.add_argument('-t', '--title', help='The title for the plot. (Required type: %(type)s, default value: %(default)s).',
                        type=str, default='', required=False)
    args = parser.parse_args()

    columnsToPlot = None if len(args.cols.split(',')) != 2 else [int(i) for i in args.cols.split(',')]
    main(args.dataset, args.output, headerPresent=args.header, separator=args.sep, classColumn=args.classCol, columnsToPlot=columnsToPlot, title=args.title)