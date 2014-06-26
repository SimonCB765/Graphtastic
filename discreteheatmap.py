import matplotlib.patches as patches
import matplotlib.path as path
import matplotlib.pyplot as plt
import numpy as np
import pandas


import scatter


def main(xCoords, yCoords, zValues, outputLocation=None, currentFigure=None, levels=None, boundary=False, boundaryColor='black', boundaryWidth=2, fill=0,
         fillAlpha=1.0, dotSize=10, colorSet='set2', colorMapping=None, title='', xLabel='', yLabel='', spinesToRemove=['right', 'top'], legend=True):
    """Produces a discretised heatmap with optional border lines drawn between areas of different values.

    The X and Y coordinate values should be supplied such that the smallest coordinates are at index [0, 0] and the largest at [-1, -1] (as would be
    returned by calling np.arange followed by np.meshgrid).

    :param xCoords:                 The x coordinates where the class values have been evaluated.
    :type xCoords:                  2 dimensional numpy array
    :param yCoords:                 The y coordinates where the class values have been evaluated.
    :type yCoords:                  2 dimensional numpy array
    :param zValues:                 The z value for each (x,y) pair.
    :type zValues:                  2 dimensional numpy array
    :param outputLocation:          The location where the figure will be saved.
    :type outputLocation:           str (or None if saving is not desired)
    :param currentFigure:           The figure from which the axes to plot the color mesh on will be taken. If not provided, then a new figure will be created.
    :type currentFigure:            matplotlib.figure.Figure
    :param levels:                  The levels at which to discretise the Z values.
    :type levels:                   extendable list like object (or None if the Z values are already discretised)
    :param boundary:                Whether the boundary demarcations should be plotted.
    :type boundary:                 boolean
    :param boundaryColor:           The color for the boundary demarcations.
    :type boundaryColor:            any value that is acceptable input for the edgecolor value of matplotlib.patches.PathPatch
    :param boundaryWidth:           The width of the lines that demarcate the boundary.
    :type boundaryWidth:            int
    :param fill:                    Whether there should be any filling performed. 0 for no filling, 1 for a dot at each (x,y) pair and 2 for full fill.
    :type fill:                     int
    :param fillAlpha:               The alpha value for the background fill.
    :type fillAlpha:                float between 0 and 1
    :param dotSize:                 The size of the dots that will be drawn on the (x,y) points. Only used if fill == 1.
    :type dotSize:                  int
    :param colorSet:                The color set to use in plotting the points. If colorMapping is provided this parameter is ignored. Otherwise, the
                                    color set will be cycled through to assign colors to distinct Z values (so should have at least as many colors as there
                                    are distinct Z values).
    :type colorSet:                 any key in the colors.colorMaps dictionary
    :param colorMapping:            A mapping from distinct Z values to their RGB color value.
    :type colorMapping:             dict
    :param title:                   The title for the plot.
    :type title:                    str
    :param xLabel:                  The label for the x axis.
    :type xLabel:                   str
    :param yLabel:                  The label for the y axis.
    :type yLabel:                   str
    :param spinesToRemove:          The spines that should be removed from the axes.
    :type spinesToRemove:           list containing any of ['left', 'right', 'top', 'bottom']
    :param legend:                  Whether a legend should be added.
    :type legend:                   boolean
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

    # Map the Z values to colors. If there are more distinct Z values than colors, then multiple Z values will be mapped to the same color.
    uniqueZValues = sorted(np.unique(zValues))
    if not colorMapping:
        colorsToUse = colors.colorMaps[colorSet]
        numberOfColors = len(colorsToUse)
        colorMapping = {}
        for i, j in enumerate(uniqueZValues):
            colorMapping[j] = colorsToUse[i % numberOfColors]

    # Discretise the Z values based on the desired levels.
    if levels:
        minZValue = zValues.min()
        maxZValue = zValues.max()
        extendedLevels = [minZValue - 1] + levels + [maxZValue + 1]  # Extend the levels with the extra start and stop points.
        enumeratedLevels = [(i,j) for i,j in enumerate(zip(extendedLevels[:-1], extendedLevels[1:]))]
        zValues = sum([(j[0] < zValues) * (zValues <= j[1]) * (i + 1) for i,j in enumeratedLevels])

    # Determine some useful statistics about the input matrices.
    numberOfRows = zValues.shape[0]
    numberOfCols = zValues.shape[1]
    halfDeltaX = abs(xCoords[0,0] - xCoords[0,1]) / 2  # Half the distance between adjacent X coordinate values.
    halfDeltaY = abs(yCoords[0,0] - yCoords[1,0]) / 2  # Half the distance between adjacent Y coordinate values.

    # Determine which adjacent row values and column values are not equal (i.e. whether xCoords[i,j] == xCoords[i,j+1] and yCoords[i,j] == yCoords[i+1,j]).
    # The test will be for whether each entry is equal to the value to its right (for row ==) or below it (for column ==).
    # If rowsNotEqual[i,j] is True, then zValues[i,j] != zValues[i,j+1]. If columnsNotEqual[i,j] is True, then zValues[i,j] != zValues[i+1,j].
    # The rightmost column of rowsNotEqual and the bottom row of columnsNotEqual will contain only False values due to the extension of the zValues
    # by duplicating the rightmost column or bottom row respectively. This ensures that both rowsNotEqual and columnsNotEqual have the same shape (otherwise
    # rowsNotEqual would have one more row and one less column than columnsNotEqual).
    zValuesExtendedColumn = np.concatenate([zValues, zValues[:, -1:]], 1)  # Duplicate the last column of the Z values.
    zValuesExtendedRow = np.concatenate([zValues, zValues[-1:, :]], 0)  # Duplicate the bottom row of the Z values.
    rowsNotEqual = np.array([i != j for i,j in zip(zValuesExtendedColumn[:, :-1], zValuesExtendedColumn[:, 1:])])
    columnsNotEqual = np.array([i != j for i,j in zip(zValuesExtendedRow[:-1, :], zValuesExtendedRow[1:, :])])

    # The boundaries, and therefore patches representing regions of color, are determined by dividing the zValues into squares. For each square, rowsNotEqual
    # and columnsNotEqual are used to determine which corners of the square have unequal Z values, and therefore where the boundary should be placed within
    # the square. The values of the corners of the square are zValues[i,i] (top left corner), zValues[i,i+1] (bottom left corner),
    # zValues[i+1,i+1] (bottom right corner) and zValues[i+1,i] (top right corner). Due the ordering of the xCoords and yCoords, the top left corner of the
    # square represents the (x,y) point with the smallest X and Y coordinate values. When plotted the square will therefore look as follows:
    # zValues[i,i+1] b zValues[i+1,i+1]
    #     a                c
    # zValues[i,i]   d zValues[i+1,i]
    # with a, b, c and d representing he point midway along the sides.
    # The relationship between the corners is determined using rowsNotEqual and columnsNotEqual and is arrange in a row vector such that the first
    # element indicates whether zValues[i,i] == zValues[i,i+1], the second whether zValues[i,i+1] == zValues[i+1,i+1], the third whether
    # zValues[i+1,i+1] == zValues[i+1,i] and the fourth whether zValues[i+1,i] == zValues[i,i].
    # This boolean row vector for the square is then converted to a row vector that contains the midpoint of the sides where the two corners at the end
    # of the side do not have equal zValues. For example, if zValues[i,i] != zValues[i,i+1] then mid point a is in the row vector. This row vector
    # indicates the points where the boundaries that go through the squares enter and exit. Each point is both an entry and exit.
    # The squares in there vectorised form are computed as:
    # [np.array([columnsNotEqual[i, j], rowsNotEqual[i+1, j], columnsNotEqual[i, j+1], rowsNotEqual[i, j]]) for i in range(numberOfRows - 1) for j in range(numberOfCols - 1)]
    # but to avoid looping through the matrices twice, the vector of the midpoints of sides with different corner Z values are computed as the
    # squares are determined.
    # The coordinates of the boundary entries and exits are represented as a set of tuples of tuples. Each external tup represents a square, and each
    # internal tuple is an entry/exit point.
    boundaryCoords = set([tuple(map(tuple, np.array([(xCoords[i,j], yCoords[i,j] + halfDeltaY), (xCoords[i+1, j] + halfDeltaX, yCoords[i+1, j]), (xCoords[i, j+1], yCoords[i, j+1] + halfDeltaY), (xCoords[i, j] + halfDeltaX, yCoords[i, j])])[np.array([columnsNotEqual[i, j], rowsNotEqual[i+1, j], columnsNotEqual[i, j+1], rowsNotEqual[i, j]])])) for i in range(numberOfRows - 1) for j in range(numberOfCols - 1)])
    boundaryCoords -= set([()])  # Remove empty tuple. A square with all four corners equal is an empty tuple (no boundary goes through it)

    # Compute the paths that represent the boundaries and the outsides of the enclosed areas of a specific set of Z values.
    paths = {}  # Dictionary indexed by the starting points, s, of boundary paths with the value associated with each starting point being a list of the
                # vertices through which that boundary passes. Tells you the entire path given its starting point.
    starts = {} # Dictionary indexed by the starting points, s, of boundary paths with the value associated with each starting point being the end point, e,
                # of the path that starts at s. Tells you the endpoint of a path given its starting point.
    ends = {}   # Dictionary indexed by the ending points, e, of boundary paths with the value associated with each ending point being the start point, s,
                # of the path that ends at e. Tells you the starting point of a path given its ending point.
    for i in boundaryCoords:
        # Loop through the squares and add any boundary segments that go through them.
        numberOfBoundaryPoints = len(i)
        i += (i[0],)  # Add the first boundary point of the square to the end of the vector. This ensures that there will correctly be a boundary edge from
                      # the last boundary entry/exit point (going counter-clockwise) to the first.
        extendStarts = set([])  # Boundary segments to add that extend the start of an already existing boundary.
        extendEnds = set([])  # Boundary segments to add that extend the end of an already existing boundary.
        fillGaps = set([])  # Boundary segments to add that extend both the start of one already existing boundary and the end of another, and therefore
                            # that fill in a small gap and complete an existing boundary.
        newAlones = set([])  # Boundary segments to add that do not extend the start or end of an already existing boundary.
        for j, k in zip(i[:-1], i[1:]):
            toAdd = set([(j,k)])  # The tuple of vertices that represent the current boundary entry/exit point.
            extendEnd = j in ends  # True if the start point of the square's boundary segment being checked is the end of an already existing boundary path.
            extendStart = k in starts  # True if the end point of the square's boundary segment being checked is the start of an already existing boundary path.
            if not (extendStart or extendEnd):
                # The new boundary segment does not extend any existing boundary segments.
                newAlones |= toAdd
            elif (extendStart and extendEnd):
                # The new boundary segment fills in a gap between two existing boundary segments.
                fillGaps |= toAdd
            else:
                if extendStart:
                    # The new boundary segment only extends the start of an existing boundary segments.
                    extendStarts |= toAdd
                if extendEnd:
                    # The new boundary segment only extends the end of an existing boundary segments.
                    extendEnds |= toAdd

        middleNeeded = numberOfBoundaryPoints > 2  # Whether you need to use the mid point of the square as one of the vertices. This is only required for
                                                   # squares where there are 3 or 4 boundary exit and entry points.
        if middleNeeded:
            middleOfSquareX = sum([j[0] for j in i[:-1]]) / numberOfBoundaryPoints  # X coordinate for the middle of the square.
            middleOfSquareY = sum([j[1] for j in i[:-1]]) / numberOfBoundaryPoints  # Y coordinate for the middle of the square.
            middleOfSquare = (middleOfSquareX, middleOfSquareY)
            for j,k in fillGaps:
                # Get the point where the existing boundary that starts at k (the end of the current square's segment) ends.
                endOfBoundaryStartingAtK = starts[k]
                del starts[k]
                # Get the point where the existing boundary that ends at j (the start of the current square's segment) starts.
                startOfBoundaryEndingAtJ = ends[j]
                del ends[j]
                # The existing boundary that ends at j now ends at the end of the existing boundary that started at k.
                starts[startOfBoundaryEndingAtJ] = endOfBoundaryStartingAtK
                ends[endOfBoundaryStartingAtK] = startOfBoundaryEndingAtJ
                # Update the path to show that the gap has been filled.
                startPath = paths[startOfBoundaryEndingAtJ]
                startPath.append(middleOfSquare)
                startPath.extend(paths[k])
                if endOfBoundaryStartingAtK != j:
                    # Only delete the path starting at k if it does not end at j. If the path starting at k does end at j, then you have a loop and
                    # paths[k] will delete the whole loop.
                    del paths[k]
            for j,k in extendStarts:
                # Get the point where the existing boundary that starts at k (the end of the current square's segment) ends.
                endOfBoundaryStartingAtK = starts[k]
                del starts[k]
                # The existing boundary that started at k now starts at j.
                starts[j] = endOfBoundaryStartingAtK
                ends[endOfBoundaryStartingAtK] = j
                # Update the paths to reflect that j is the new start point.
                paths[j] = [j, middleOfSquare] + paths[k]
                del paths[k]
            for j,k in extendEnds:
                # Get the point where the existing boundary that ends at j (the start of the current square's segment) starts.
                startOfBoundaryEndingAtJ = ends[j]
                del ends[j]
                # The existing boundary that ends at j now ends at k.
                ends[k] = startOfBoundaryEndingAtJ
                starts[startOfBoundaryEndingAtJ] = k
                # Update the paths to reflect that k is the new end point.
                paths[startOfBoundaryEndingAtJ].extend([middleOfSquare, k])
            for j,k in newAlones:
                # Add the new unconnected boundary segment.
                paths[j] = [j,middleOfSquare,k]
                starts[j] = k
                ends[k] = j
        else:
            for j,k in fillGaps:
                # Get the point where the existing boundary that starts at k (the end of the current square's segment) ends.
                endOfBoundaryStartingAtK = starts[k]
                del starts[k]
                # Get the point where the existing boundary that ends at j (the start of the current square's segment) starts.
                startOfBoundaryEndingAtJ = ends[j]
                del ends[j]
                # The existing boundary that ends at j now ends at the end of the existing boundary that started at k.
                starts[startOfBoundaryEndingAtJ] = endOfBoundaryStartingAtK
                ends[endOfBoundaryStartingAtK] = startOfBoundaryEndingAtJ
                # Update the path to show that the gap has been filled.
                startPath = paths[startOfBoundaryEndingAtJ]
                startPath.extend(paths[k])
                if endOfBoundaryStartingAtK != j:
                    # Only delete the path starting at k if it does not end at j. If the path starting at k does end at j, then you have a loop and
                    # paths[k] will delete the whole loop.
                    del paths[k]
            for j,k in extendStarts:
                # Get the point where the existing boundary that starts at k (the end of the current square's segment) ends.
                endOfBoundaryStartingAtK = starts[k]
                del starts[k]
                # The existing boundary that started at k now starts at j.
                starts[j] = endOfBoundaryStartingAtK
                ends[endOfBoundaryStartingAtK] = j
                # Update the paths to reflect that j is the new start point.
                paths[j] = [j] + paths[k]
                del paths[k]
            for j,k in extendEnds:
                # Get the point where the existing boundary that ends at j (the start of the current square's segment) starts.
                startOfBoundaryEndingAtJ = ends[j]
                del ends[j]
                # The existing boundary that ends at j now ends at k.
                ends[k] = startOfBoundaryEndingAtJ
                starts[startOfBoundaryEndingAtJ] = k
                # Update the paths to reflect that k is the new end point.
                paths[startOfBoundaryEndingAtJ].append(k)
            for j,k in newAlones:
                # Add the new unconnected boundary segment.
                paths[j] = [j,k]
                starts[j] = k
                ends[k] = j

    if fill == 1:
        # Place the color dots on the graph.
        xValuesForDots = pandas.DataFrame([j for i in xCoords for j in i])
        yValuesForDots = pandas.DataFrame([j for i in yCoords for j in i])
        zValuesForDots = pandas.Series([j for i in zValues for j in i])
        scatter.plot(xValuesForDots, yValuesForDots, classLabels=zValuesForDots, currentFigure=currentFigure, size=dotSize, edgeColor='none',
                     title=title, xLabel=xLabel, yLabel=yLabel, faceColorSet=colorSet, linewidths=0, alpha=fillAlpha, legend=legend)
    elif fill == 2:
        # Fill the regions with solid colors.
        pass

    if boundary:
        for i in paths:
            verts = paths[i]
            codes = [path.Path.MOVETO] + ([path.Path.LINETO] * (len(verts) - 1))
            boundary = path.Path(verts, codes)
            patch = patches.PathPatch(boundary, facecolor='none', linewidth=boundaryWidth, edgecolor=boundaryColor, alpha=1, linestyle='solid')
            axes.add_patch(patch)

    if outputLocation:
        plt.savefig(outputLocation, bbox_inches='tight', transparent=True)
    else:
        return currentFigure, axes