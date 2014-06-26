import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import RadioButtons, Button
from multiprocessing import Pool
import numpy as np
import pandas
import scipy.spatial

import nearestneighbours

# Import the scatterplot script, colors and the script for generating the color fill.
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, parent_dir)
import colors
import discreteheatmap
import scatter


class InteractiveNNDemo:
    """An interactive nearest neighbours demonstration."""

    def __init__(self, datasetLocation, neighbours=3, headerPresent=False, separator='\t', classColumn=-1, columnsToPlot=None, title='',
                 divisions=100, radius=0.025, colorSet='set2', poolSize=5):
        """Create a scatter plot along with the NN decision boundaries induced by the data.

        :param datasetLocation:     The location of the dataset to generate the plot from. The class is assumed to be integer type.
        :type datasetLocation:      str
        :param neighbours:          The number of neighbours to use in the classification.
        :type neighbours:           int
        :param headerPresent:       Whether a single line header is present in the dataset file.
        :type headerPresent:        boolean
        :param separator:           The string that separates values in the file containing dataset.
        :type separator:            str
        :param classColumn:         The index of the column containing the class of the observations (can use negative indexing). The class is assumed to be integer type.
        :type classColumn:          int
        :param columnsToPlot:       The indices of the two columns to plot (defaults to the first and second columns [0, 1]).
        :type columnsToPlot:        list of ints
        :param title:               The title for the figure.
        :type title:                str
        :param divisions:           The number of slices to divide each axis into. Each of the divisions^2 (x,y) pairs will be evaluated for its class.
        :type divisions:            int
        :param radius:              The radius of the plotted points.
        :type radius:               float
        :param colorSet:            The color set to use in plotting the points. Will be cycled through (so should have at least as many colors as there are classes).
        :type colorSet:             any key in the colors.colorMaps dictionary
        :param poolSize:            The number of workers to create in the multiprocessing.Pool process pool.
        :type poolSize:             int

        """

        self.plottedPoints = {}
        self.neighbours = neighbours
        self.title = title
        self.divisions = divisions
        self.pointRadius = radius
        self.poolSize = poolSize
        self.currentlyDeleting = True

        # Extract the original dataset.
        if not columnsToPlot or len(columnsToPlot) != 2:
            # If two columns were not specified, then default to plotting the first two columns.
            columnsToPlot = [0, 1]
        originalDataset = pandas.read_csv(datasetLocation, sep=separator, header=(0 if headerPresent else None))
        self.originalDataset = originalDataset.iloc[:, columnsToPlot + [-1]]
        self.plottedDataset = self.originalDataset.copy(deep=True)

        # Map the classes to colors.
        uniqueClasses = sorted(self.originalDataset.iloc[:, -1].unique())
        colorsToUse = colors.colorMaps[colorSet]
        numberOfColors = len(colorsToUse)
        self.classToColorMapping = {}
        for i, j in enumerate(uniqueClasses):
            self.classToColorMapping[j] = colorsToUse[i % numberOfColors]
        self.currentAdditionClass = uniqueClasses[0]

        # Create the NN classifier.
        self.classifier = nearestneighbours.NearestNeighbours(self.plottedDataset)

        # Create the figure.
        self.currentFigure = plt.figure()
        self.axes = self.currentFigure.add_subplot(1, 1, 1)


    def on_class_change(self, label):
        self.currentAdditionClass = int(label)


    def on_click(self, event):
        if event.inaxes == self.axes:
            # Only do something if the click occurred within the axes containing the plot.
            coords = (event.xdata, event.ydata)
            if self.currentlyDeleting:
                # If the deletion radio button is filled in.
                deletionFound = False
                for i in self.plottedPoints:
                    if scipy.spatial.distance.euclidean(i, coords) < self.pointRadius:
                        # Click was inside a point.
                        deletionFound = True
                        break
                if deletionFound:
                    self.plottedPoints[i].remove()
                    del self.plottedPoints[i]
                    plt.draw()
            else:
                # A new point should be added to the plot at the mouse coordinates.
                datapoint = patches.Circle(coords, self.pointRadius, facecolor=self.classToColorMapping[self.currentAdditionClass], edgecolor='black', linewidth=1, alpha=0.75)
                self.plottedPoints[coords] = datapoint
                self.axes.add_patch(datapoint)
                plt.draw()


    def on_delete_change(self, label):
        if label == 'Add':
            self.currentlyDeleting = False
        else:
            self.currentlyDeleting = True


    def on_recompute(self, event):
        print('Recompute')


    def on_reset(self, event):
        print('Reset')


    def save(self, outputLocation):
        self.currentFigure.savefig(outputLocation, bbox_inches='tight', transparent=True)


    def start(self):
        """Generate the plot.
        """

        # Extract the data to plot.
        featureOne = self.originalDataset.iloc[:, columnsToPlot[0]]
        featureTwo = self.originalDataset.iloc[:, columnsToPlot[1]]
        classes = self.originalDataset.iloc[:, -1]

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
        featureOneDelta = (featureOneMax - featureOneMin) / self.divisions
        featureTwoDelta = (featureTwoMax - featureTwoMin) / self.divisions
        featureOneSteps = np.arange(featureOneMin, featureOneMax + featureOneDelta, featureOneDelta)
        featureTwoSteps = np.arange(featureTwoMin, featureTwoMax + featureTwoDelta, featureTwoDelta)
        featureOneMesh, featureTwoMesh = np.meshgrid(featureOneSteps, featureTwoSteps)

        # Classify the points on the mesh.
        workerPool = Pool(self.poolSize)
        classificatons = workerPool.map(worker, [(self.classifier, pandas.DataFrame([featureOneMesh[:, i], featureTwoMesh[:, i]]).T, self.neighbours) for i in range(len(featureOneSteps))])

        # Draw the boundary and classification regions. Ideally pcolormesh would be used, but with alpha values this gives unsightly lines along the edges of the
        # mesh due to overlapping squares (http://matplotlib.1069221.n5.nabble.com/Quadmesh-with-alpha-without-the-nasty-edge-effects-td41039.html) that
        # edgecolor='none' does not fix. Instead use my workaround that plays nice with alpha values and gives boundary lines.
        self.currentFigure, self.axes = discreteheatmap.main(featureOneMesh, featureTwoMesh, np.transpose(np.array(classificatons)),
                                                             currentFigure=self.currentFigure, boundary=True, boundaryColor='black', boundaryWidth=2,
                                                             fill=1, fillAlpha=0.75, dotSize=200/self.divisions, colorMapping=self.classToColorMapping,
                                                             title=self.title, xLabel=self.originalDataset.columns[0],
                                                             yLabel=self.originalDataset.columns[1], legend=False)

        # Plot the data.
        for index, series in self.originalDataset.iterrows():
            observationClass = series[-1]
            pointLocation = (series[0], series[1])
            datapoint = patches.Circle(pointLocation, self.pointRadius, facecolor=self.classToColorMapping[observationClass], edgecolor='black', linewidth=1, alpha=0.75)
            self.plottedPoints[pointLocation] = datapoint
            self.axes.add_patch(datapoint)

        # Plot a fake scatter plot to get the legend working nicely.
        legendPoints = []
        for i in sorted(self.classToColorMapping):
            scattered = self.axes.scatter(featureOne[classes == i], featureTwo[classes == i], s=40, c=self.classToColorMapping[i], label=str(i), marker='o', edgecolor='black', linewidths=0.25, alpha=0.75)
            legendPoints.append(scattered)
            scattered.remove()

        # Add a legend.
        legend = self.axes.legend(legendPoints, [str(i) for i in sorted(self.classToColorMapping)], bbox_to_anchor=(1.05, 0.5), loc=6, borderaxespad=0, frameon=True, scatterpoints=1)
        legendFrame = legend.get_frame()
        legendFrame.set_facecolor('white')
        legendFrame.set_edgecolor('black')
        legendFrame.set_linewidth(0.2)
        for i in legend.get_texts():
            i.set_color('0.25')

        # Reset the axes to ensure that the color mesh plotting hasn't changed it, which would cause a bunch of whitespace around the edges of the plot.
        plt.axis('scaled')
        axisOnePadding = featureOneRange * 0.05
        self.axes.set_xlim([featureOneMin - axisOnePadding, featureOneMax + axisOnePadding])
        axisTwoPadding = featureTwoRange * 0.05
        self.axes.set_ylim([featureTwoMin - axisTwoPadding, featureTwoMax + axisTwoPadding])

        # Create the recompute boundaries button.
        recomputeAxesLoc = plt.axes([0.05, 0.54, 0.07, 0.075], axisbg='white')
        recomputeButton = Button(recomputeAxesLoc, 'Recompute')
        recomputeButton.on_clicked(self.on_recompute)

        # Create the reset button.
        resetAxesLoc = plt.axes([0.05, 0.44, 0.07, 0.075], axisbg='white')
        resetButton = Button(resetAxesLoc, 'Reset')
        resetButton.on_clicked(self.on_reset)

        # Create the add/delete radio button.
        addDeleteAxesLoc = plt.axes([0.05, 0.85, 0.07, 0.1], axisbg='0.85')
        addDeleteAxesLoc.set_title('Add/Delete Points')
        addDeleteRadio = RadioButtons(addDeleteAxesLoc, active=1, activecolor='black', labels=['Add', 'Delete'])
        addDeleteRadio.on_clicked(self.on_delete_change)

        # Create the class radio button.
        classAxesLoc = plt.axes([0.05, 0.05, 0.07, 0.025 * len(self.classToColorMapping)], axisbg='0.85')
        classAxesLoc.set_title('Class Options')
        classRadio = RadioButtons(classAxesLoc, active=0, activecolor='black', labels=[i for i in sorted(self.classToColorMapping)])
        classRadio.on_clicked(self.on_class_change)

        # Attach the mouse click event.
        cid = self.currentFigure.canvas.mpl_connect('button_press_event', self.on_click)

        # Display the figure maximised. These commands to maximise the figure are for the Qt4Agg backend. If a different backend is being used, then
        # the maximisation command will likely need to be changed.
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.show()

        # Disconnect the mouse click event.
        self.currentFigure.canvas.mpl_disconnect(cid)


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
    parser.add_argument('-n', '--neighbours', help='The number k of neighbours to use. (Required type: %(type)s, default value: %(default)s).',
                        type=int, default=3, required=False)
    parser.add_argument('-f', '--header', help='Whether a header is present in dataset file. (Default value: No header).',
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
                        type=int, default=100, required=False)
    parser.add_argument('-r', '--radius', help='The radius of the plotted points. (Required type: %(type)s, default value: %(default)s).',
                        type=float, default=0.025, required=False)
    parser.add_argument('-o', '--output', help='The location to save the final modified image. (Required type: %(type)s, default value: %(default)s).',
                        type=str, default=None, required=False)
    args = parser.parse_args()

    columnsToPlot = None if len(args.cols.split(',')) != 2 else [int(i) for i in args.cols.split(',')]
    demo = InteractiveNNDemo(args.dataset, args.neighbours, headerPresent=args.header, separator=args.sep, classColumn=args.classCol, columnsToPlot=columnsToPlot, title=args.title, divisions=args.sectors, radius=args.radius)
    demo.start()
    if args.output:
        demo.save(args.output)