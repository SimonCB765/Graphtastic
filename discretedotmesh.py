import matplotlib.pyplot as plt
from multiprocessing import Pool
import numpy as np
import pandas

import colors
import scatter


def main(xCoords, yCoords, classValues, outputLocation=None, currentFigure=None, border=False, colorSet='set2', classToColorMapping=None, borderColor='0.25',
         boundaryWidth=2, poolSize=5, title='', xLabel='', yLabel='', spinesToRemove=['right', 'top'], dotSize=10, legend=True, dotAlpha=1.0):
    """Produces a dotted color image where colors are based on discrete class values.

    The X and Y coordinate values should be supplied such that the smallest coordinates are at index [0, 0] and the largest at [-1, -1] (as would be
    returned by calling np.arange followed by np.meshgrid).

    :param xCoords:                 The x coordinates where the class values have been evaluated.
    :type xCoords:                  2 dimensional numpy array
    :param yCoords:                 The y coordinates where the class values have been evaluated.
    :type yCoords:                  2 dimensional numpy array
    :param classValues:             The class value for each (x,y) pair.
    :type classValues:              2 dimensional numpy array
    :param outputLocation:          The location where the figure will be saved.
    :type outputLocation:           str (or None if saving is not desired)
    :param currentFigure:           The figure from which the axes to plot the color mesh on will be taken. If not provided, then a new figure will be created.
    :type currentFigure:            matplotlib.figure.Figure
    :param border:                  Whether the boundary demarcations should be plotted.
    :type border:                   boolean
    :param colorSet:                The color set to use in plotting the points. If classToColorMapping is provided this parameter is ignored. Otherwise, the
                                    color set will be cycled through to assign colors to classes (so should have at least as many colors as there are classes).
    :type colorSet:                 any key in the colors.colorMaps dictionary
    :param classToColorMapping:     A mapping form class names to their RGB color value.
    :type classToColorMapping:      dict
    :param borderColor:             The color for the boundary demarcations.
    :type borderColor:              any value that is acceptable input for the color value of matplotlib.pyplot.plot
    :param boundaryWidth:           The width of the lines that demarcate the boundary.
    :type boundaryWidth:            int
    :param poolSize:                The number of workers to create in the multiprocessing.Pool process pool.
    :type poolSize:                 int
    :param title:                   The title for the plot.
    :type title:                    str
    :param xLabel:                  The label for the x axis.
    :type xLabel:                   str
    :param yLabel:                  The label for the y axis.
    :type yLabel:                   str
    :param spinesToRemove:          The spines that should be removed from the axes.
    :type spinesToRemove:           list containing any of ['left', 'right', 'top', 'bottom']
    :param dotSize:                 The size of the points on the mesh points.
    :type dotSize:                  int
    :param legend:                  Whether a legend should be added.
    :type legend:                   boolean
    :param dotAlpha:                The alpha value for the background dots.
    :type dotAlpha:                 float between 0 and 1
    :returns :                      The figure and axes on which the color mesh was plotted if saving is not to be performed.
    :type :                         objects of type matplotlib.figure.Figure, matplotlib.axes.Axes

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

    # Map the classes to colors. If there are more classes than colors, then multiple classes will be mapped to the same color.
    classes = sorted(np.unique(classValues))
    if not classToColorMapping:
        colorsToUse = colors.colorMaps[colorSet]
        numberOfColors = len(colorsToUse)
        classToColorMapping = {}
        for i, j in enumerate(classes):
            classToColorMapping[j] = colorsToUse[i % numberOfColors]

    # Determine the differences in coordinates of adjacent points on the mesh.
    deltaX = (xCoords[0, 1] - xCoords[0, 0])
    halfDeltaX = deltaX / 2
    deltaY = (yCoords[1, 0] - yCoords[0, 0])
    halfDeltaY = deltaY / 2

    # Create the boundaries.
    if border:
        # Determine horizontal and vertical boundaries.
        workerPool = Pool(poolSize)
        verticalBoundaries = workerPool.map(boundary_calculater, classValues)
        for i, j in enumerate(verticalBoundaries):
            # J contains all row indices, k, such that there needs to be a boundary between between the kth and k+1th x coordinate. The value of i
            # indicates which y coordinates should be used as the top and bottom of the boundary line.
            xVals = xCoords[0, j] + halfDeltaX  # X boundary value is half way between kth and k+1th x coordinates.
            yMin = yCoords[i, 0] - halfDeltaY  # The starting y coordinate is half way between the y coordinates stored in row i and the one in i-1 (or just
                                               # half way below the one in i if i is 0).
            yMax = yCoords[i, 0] + halfDeltaY  # The final y coordinate is half way between the y coordinates stored in row i and the one in i+1 (or just
                                               # half way above the one in i if i indexes the top row).
            for k in xVals:
                plt.plot([k, k], [yMin, yMax], c=borderColor, linewidth=boundaryWidth)
        horizontalBoundaries = workerPool.map(boundary_calculater, classValues.T)
        for i, j in enumerate(horizontalBoundaries):
            # J contains all column indices, k, such that there needs to be a boundary between between the kth and k+1th y coordinate. The value of i
            # indicates which x coordinates should be used as the top and bottom of the boundary line.
            xMin = xCoords[0, i] - halfDeltaX  # The starting x coordinate is half way between the x coordinates stored in column i and the one in i-1
                                               # (or just half way to the left of the one in i if i is 0).
            xMax = xCoords[0, i] + halfDeltaX  # The final x coordinate is half way between the x coordinates stored in column i and the one in i+1 (or just
                                               # half way to the right of the one in i if i indexes the rightmost column).
            yVals = yCoords[j, 0] + halfDeltaY  # Y boundary value is half way between kth and k+1th y coordinates.
            for k in yVals:
                plt.plot([xMin, xMax], [k, k], c=borderColor, linewidth=boundaryWidth)

    # Place the color dots on the graph.
    xValuesForDots = pandas.DataFrame([j for i in xCoords for j in i])
    yValuesForDots = pandas.DataFrame([j for i in yCoords for j in i])
    classValuesForDots = pandas.Series([j for i in classValues for j in i])
    scatter.plot(xValuesForDots, yValuesForDots, classLabels=classValuesForDots, currentFigure=currentFigure, size=dotSize, edgeColor='none',
                 title=title, xLabel=xLabel, yLabel=yLabel, faceColorSet=colorSet, linewidths=0, alpha=dotAlpha, legend=legend)

    if outputLocation:
        plt.savefig(outputLocation, bbox_inches='tight', transparent=True)
    else:
        return currentFigure, axes


def boundary_calculater(row):
    """Function to determine all indices, i, in a row where the class at row[i+1] != the class of row[i].

    Auxiliary function is required to meet restrictions placed on Pool.map.

    :param row:     A row of class values.
    :type row:      numpy array
    :returns :      The indices, i, where row[i] != row[i+1].
    :type :         list of ints


    """
    return [i for i,j in enumerate(row[:-1]) if j != row[i + 1]]