import matplotlib.pyplot as plt
from multiprocessing import Pool
import numpy as np

import colors


def main(xCoords, yCoords, classValues, outputLocation=None, currentFigure=None, border=False, colorSet='set2', classToColorMapping=None, borderColor='0.25',
         fillAlpha=1.0, boundaryWidth=2, poolSize=5):
    """
    X and Y values are supplied such that the smallest values are at [0, 0] and larges at [-1, -1] (as would be returned by normal calling of np.arange
    followed by np.meshgrid

    colorSet is ignored if classToColorMapping is supplied

    edgecolor='none' still occasionally fails ad you will get stupid darker lines in between patches. If this happens just rerun the code and it should get fixed.
    """

    # Get the axes the will be used for the plot.
    try:
        axes = currentFigure.gca()
    except AttributeError:
        # If the figure is not given then create it.
        currentFigure = plt.figure()
        axes = currentFigure.add_subplot(1, 1, 1)

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
            xVals = xCoords[0, j] + halfDeltaX
            yMin = yCoords[i, 0] - halfDeltaY
            yMax = yCoords[i, 0] + halfDeltaY
            for k in xVals:
                plt.plot([k, k], [yMin, yMax], c=borderColor, linewidth=boundaryWidth)
        horizontalBoundaries = workerPool.map(boundary_calculater, classValues.T)
        for i, j in enumerate(horizontalBoundaries):
            xMin = xCoords[0, i] - halfDeltaX
            xMax = xCoords[0, i] + halfDeltaX
            yVals = yCoords[j, 0] + halfDeltaY
            for k in yVals:
                plt.plot([xMin, xMax], [k, k], c=borderColor, linewidth=boundaryWidth)

    # Shift the data to make the filling with color work.
    xAddition = xCoords[:, 0:1] - deltaX
    xCoords = np.concatenate((xAddition, (np.repeat(xCoords, 2, axis=1))[:, :-1]), axis=1) + halfDeltaX
    yCoords = np.repeat(yCoords, 2, axis=1) - halfDeltaY
    classValues = np.repeat(classValues, 2, axis=1)

    # Fill in the color. Shifting the y values very slightly causes the fills to imperceptibly overlap. This doesn't introduce visible overlaps, but does
    # cause any potential white line artefacts (likely introduced due to scales and float division used to determine grid coordinates) to be prevented.
    for i, j, k in zip(xCoords, yCoords, classValues):
        yMin = j[0]
        yMax = yMin + (deltaY * 0.99)
        yMin -= (deltaY * 0.01)
        for c in classes:
            axes.fill_between(i, yMin, yMax, where=(k==c), color=classToColorMapping[c], alpha=fillAlpha, edgecolor='none')

    if outputLocation:
        plt.savefig(outputLocation, bbox_inches='tight', transparent=True)
    else:
        return currentFigure, axes


def boundary_calculater(x):
    """
    """
    return [i for i,j in enumerate(x[:-1]) if j != x[i + 1]]