import argparse
import pandas
import matplotlib.pyplot as plt

import colors


def main(datasetLocation, outputLocation, headerPresent=False, separator='\t', classColumn=None, columnsToPlot=None, title=''):
    """

    :param datasetLocation:     The location of the dataset to generate a scatterplot from.
    :type datasetLocation:      str

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


def plot(featureOne, featureTwo, outputLocation, classLabels=pandas.DataFrame(), currentFigure=None, title='', xLabel='', yLabel='', size=40, shape='o', edgeColor='black',
		 faceColorSet='set2', linewidths=0.25, alpha=0.75, spinesToRemove=['top', 'right']):
    """
    """

    # Get the axes the will be used for the plot.
    try:
        axes = currentFigure.gca()
    except AttributeError:
        # If the figure is not given then create it.
        currentFigure = plt.figure()
        axes = currentFigure.add_subplot(1, 1, 1)

    # Remove certain spines.
    for i in spinesToRemove:
        axes.spines[i].set_visible(False)

    # Slightly alter remaining spines.
    for i in set(['left', 'right', 'top', 'bottom']) - set(spinesToRemove):
        axes.spines[i].set_linewidth(0.75)
        axes.spines[i].set_color('0.25')

    # Alter ticks.
    axes.xaxis.set_ticks_position('none')
    for i in axes.xaxis.get_ticklabels():
        i.set_color('0.25')
    axes.yaxis.set_ticks_position('none')
    for i in axes.yaxis.get_ticklabels():
        i.set_color('0.25')

    # Setup the title.
    axes.set_title(title, fontsize=22, color='0.25')

    # Label the axes.
    plt.xlabel(xLabel, fontsize=16, color='0.25')
    plt.ylabel(yLabel, fontsize=16, color='0.25')

    # Generate the plot.
    if classLabels.empty:
        axes.scatter(featureOne, featureTwo, s=size, c='red', marker=shape, edgecolor=edgeColor, linewidths=linewidths, alpha=alpha)
    else:
        uniqueLabels = classLabels.unique()
        for i in uniqueLabels:
            axes.scatter(featureOne[classLabels == i], featureTwo[classLabels == i], s=size, c=colors.colorMaps[faceColorSet][i], label=str(i), marker=shape, edgecolor=edgeColor, linewidths=linewidths, alpha=alpha)
        # Add a legend.
        legend = axes.legend(bbox_to_anchor=(1.05, 0.5), loc=6, borderaxespad=0, frameon=True, scatterpoints=1)
        legendFrame = legend.get_frame()
        legendFrame.set_facecolor('white')
        legendFrame.set_edgecolor('black')
        legendFrame.set_linewidth(0.2)
        for i in legend.get_texts():
            i.set_color('0.25')

    plt.savefig(outputLocation, bbox_inches='tight', transparent=True)


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