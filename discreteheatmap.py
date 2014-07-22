import matplotlib.patches as patches
import matplotlib.path as path
import matplotlib.pyplot as plt
import numpy as np
import pandas


import scatter


def main(xCoords, yCoords, zValues, outputLocation=None, currentFigure=None, levels=None, boundary=False, boundaryColor='black', boundaryWidth=2,
         boundaryStyle='solid', fill=0, fillAlpha=1.0, dotSize=10, colorSet='set2', colorMapping=None, title='', xLabel='', yLabel='',
         spinesToRemove=['right', 'top'], legend=True):
    """Produces a discretised heatmap with optional border lines drawn between areas of different values.

    The X and Y coordinate values should be supplied such that the smallest coordinates are at index [0, 0] and the largest at [-1, -1] (as would be
    returned by calling np.arange followed by np.meshgrid).

    Solid filling (fill==2) can be temperamental when the resolution of the (X,Y) grid gets very low (e.g. 10x10).

    :param xCoords:                 The x coordinates where the Z values have been evaluated.
    :type xCoords:                  2 dimensional numpy array
    :param yCoords:                 The y coordinates where the Z values have been evaluated.
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
    :param boundaryStyle:           The style of the boundary line.
    :type boundaryStyle:            any valid value for the matplotlib.patches.PathPatch linestyle parameter
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

    # Discretise the Z values based on the desired levels.
    if levels:
        minZValue = zValues.min()
        maxZValue = zValues.max()
        extendedLevels = [minZValue - 1] + levels + [maxZValue + 1]  # Extend the levels with the extra start and stop points.
        enumeratedLevels = [(i,j) for i,j in enumerate(zip(extendedLevels[:-1], extendedLevels[1:]))]
        zValues = sum([(j[0] < zValues) * (zValues <= j[1]) * (i + 1) for i,j in enumeratedLevels])

    # Map the Z values to colors. If there are more distinct Z values than colors, then multiple Z values will be mapped to the same color.
    uniqueZValues = sorted(np.unique(zValues))
    if not colorMapping:
        colorsToUse = colors.colorMaps[colorSet]
        numberOfColors = len(colorsToUse)
        colorMapping = {}
        for i, j in enumerate(uniqueZValues):
            colorMapping[j] = colorsToUse[i % numberOfColors]

    # Determine some useful statistics about the input matrices.
    numberOfRows = zValues.shape[0]
    numberOfCols = zValues.shape[1]
    xMin = xCoords.min()
    xMax = xCoords.max()
    halfDeltaX = abs(np.ediff1d([xCoords[0,0], xCoords[0,1]]))[0] / 2  # Half the distance between adjacent X coordinate values.
    yMin = yCoords.min()
    yMax = yCoords.max()
    halfDeltaY = abs(np.ediff1d([yCoords[0,0], yCoords[1,0]]))[0] / 2  # Half the distance between adjacent Y coordinate values.

    # Compute the boundaries only if they are needed.
    if boundary or fill == 2:
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
        paths = {}         # Dictionary indexed by the starting points, s, of boundary paths with the value associated with each starting point being a list of
                           # the vertices through which that boundary passes. Tells you the entire path given its starting point.
        startsToEnds = {}  # Dictionary indexed by the starting points, s, of boundary paths with the value associated with each starting point being the end
                           # point, e, of the path that starts at s. Tells you the endpoint of a path given its starting point.
        endsToStarts = {}  # Dictionary indexed by the ending points, e, of boundary paths with the value associated with each ending point being the start
                           # point, s, of the path that ends at e. Tells you the starting point of a path given its ending point.
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
                extendEnd = j in endsToStarts  # True if the start point of the square's boundary segment being checked is the end of an already existing boundary path.
                extendStart = k in startsToEnds  # True if the end point of the square's boundary segment being checked is the start of an already existing boundary path.
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
                    endOfBoundaryStartingAtK = startsToEnds[k]
                    del startsToEnds[k]
                    # Get the point where the existing boundary that ends at j (the start of the current square's segment) starts.
                    startOfBoundaryEndingAtJ = endsToStarts[j]
                    del endsToStarts[j]
                    # Update the path to show that the gap has been filled.
                    startPath = paths[startOfBoundaryEndingAtJ]
                    startPath.append(middleOfSquare)
                    if endOfBoundaryStartingAtK != j:
                        # Only delete the path starting at k if it does not end at j. If the path starting at k does end at j, then you have a loop and
                        # paths[k] will delete the whole loop.
                        startPath.extend(paths[k])
                        del paths[k]
                        # The existing boundary that ends at j now ends at the end of the existing boundary that started at k.
                        startsToEnds[startOfBoundaryEndingAtJ] = endOfBoundaryStartingAtK
                        endsToStarts[endOfBoundaryStartingAtK] = startOfBoundaryEndingAtJ
                    else:
                        # Close the loop by adding k to the end of the path from k to j, thereby making it go from k to k. As there are two paths recorded for
                        # each closed loop, do not update the dictionary of endpoints, as this will interfere with the correct recording of the second closed loop
                        # path (i.e. do not do endsToStarts[k] = k).
                        startPath.append(k)
                        startsToEnds[k] = k
                for j,k in extendStarts:
                    # Get the point where the existing boundary that starts at k (the end of the current square's segment) ends.
                    endOfBoundaryStartingAtK = startsToEnds[k]
                    del startsToEnds[k]
                    # The existing boundary that started at k now starts at j.
                    startsToEnds[j] = endOfBoundaryStartingAtK
                    endsToStarts[endOfBoundaryStartingAtK] = j
                    # Update the paths to reflect that j is the new start point.
                    paths[j] = [j, middleOfSquare] + paths[k]
                    del paths[k]
                for j,k in extendEnds:
                    # Get the point where the existing boundary that ends at j (the start of the current square's segment) starts.
                    startOfBoundaryEndingAtJ = endsToStarts[j]
                    del endsToStarts[j]
                    # The existing boundary that ends at j now ends at k.
                    endsToStarts[k] = startOfBoundaryEndingAtJ
                    startsToEnds[startOfBoundaryEndingAtJ] = k
                    # Update the paths to reflect that k is the new end point.
                    paths[startOfBoundaryEndingAtJ].extend([middleOfSquare, k])
                for j,k in newAlones:
                    # Add the new unconnected boundary segment.
                    paths[j] = [j,middleOfSquare,k]
                    startsToEnds[j] = k
                    endsToStarts[k] = j
            else:
                for j,k in fillGaps:
                    # Get the point where the existing boundary that starts at k (the end of the current square's segment) ends.
                    endOfBoundaryStartingAtK = startsToEnds[k]
                    del startsToEnds[k]
                    # Get the point where the existing boundary that ends at j (the start of the current square's segment) starts.
                    startOfBoundaryEndingAtJ = endsToStarts[j]
                    del endsToStarts[j]
                    # Update the path to show that the gap has been filled.
                    startPath = paths[startOfBoundaryEndingAtJ]
                    if endOfBoundaryStartingAtK != j:
                        # Only delete the path starting at k if it does not end at j. If the path starting at k does end at j, then you have a loop and
                        # paths[k] will delete the whole loop.
                        startPath.extend(paths[k])
                        del paths[k]
                        # The existing boundary that ends at j now ends at the end of the existing boundary that started at k.
                        startsToEnds[startOfBoundaryEndingAtJ] = endOfBoundaryStartingAtK
                        endsToStarts[endOfBoundaryStartingAtK] = startOfBoundaryEndingAtJ
                    else:
                        # Close the loop by adding k to the end of the path from k to j, thereby making it go from k to k. As there are two paths recorded for
                        # each closed loop, do not update the dictionary of endpoints, as this will interfere with the correct recording of the second closed loop
                        # path (i.e. do not do endsToStarts[k] = k).
                        startPath.append(k)
                        startsToEnds[k] = k
                for j,k in extendStarts:
                    # Get the point where the existing boundary that starts at k (the end of the current square's segment) ends.
                    endOfBoundaryStartingAtK = startsToEnds[k]
                    del startsToEnds[k]
                    # The existing boundary that started at k now starts at j.
                    startsToEnds[j] = endOfBoundaryStartingAtK
                    endsToStarts[endOfBoundaryStartingAtK] = j
                    # Update the paths to reflect that j is the new start point.
                    paths[j] = [j] + paths[k]
                    del paths[k]
                for j,k in extendEnds:
                    # Get the point where the existing boundary that ends at j (the start of the current square's segment) starts.
                    startOfBoundaryEndingAtJ = endsToStarts[j]
                    del endsToStarts[j]
                    # The existing boundary that ends at j now ends at k.
                    endsToStarts[k] = startOfBoundaryEndingAtJ
                    startsToEnds[startOfBoundaryEndingAtJ] = k
                    # Update the paths to reflect that k is the new end point.
                    paths[startOfBoundaryEndingAtJ].append(k)
                for j,k in newAlones:
                    # Add the new unconnected boundary segment.
                    paths[j] = [j,k]
                    startsToEnds[j] = k
                    endsToStarts[k] = j

    if boundary:
        # Add the boundaries if requested.
        for i in paths:
            verts = paths[i]
            codes = [path.Path.MOVETO] + ([path.Path.LINETO] * (len(verts) - 1))
            boundary = path.Path(verts, codes)
            patch = patches.PathPatch(boundary, facecolor='none', linewidth=boundaryWidth, edgecolor=boundaryColor, alpha=1, linestyle=boundaryStyle)
            axes.add_patch(patch)

    if fill == 1:
        # Place the color dots on the graph.
        xValuesForDots = pandas.DataFrame([j for i in xCoords for j in i])
        yValuesForDots = pandas.DataFrame([j for i in yCoords for j in i])
        zValuesForDots = pandas.Series([j for i in zValues for j in i])
        scatter.plot(xValuesForDots, yValuesForDots, classLabels=zValuesForDots, currentFigure=currentFigure, size=dotSize, edgeColor='none',
                     title=title, xLabel=xLabel, yLabel=yLabel, colorMapping=colorMapping, linewidths=0, alpha=fillAlpha, legend=legend)
    elif fill == 2:
        # Fill the regions with solid colors.
        pathsStartsToRegionZValues = {}  # A record of the starting points of each path to output, along with the Z value of its interior.

        # First determine the color of all already closed paths (i.e. paths that do not touch an edge of the (X, Y) grid). Each already closed path is
        # represented by a counter-clockwise and clockwise path (representing the interior and everything exterior of the closed region respectively).
        # In order to stay consistent with all other paths, any clockwise path is to be removed.
        closedPathStarts = set([i for i in startsToEnds if startsToEnds[i] == i])
        for i in closedPathStarts:
            pathIsCCW, areaZValue = test_clockwise(paths[i], xCoords, halfDeltaX, yCoords, halfDeltaY, zValues)
            if pathIsCCW:
                # The path is counter-clockwise, so record it as being kept along with its Z value.
                pathsStartsToRegionZValues[i] = areaZValue

        # Second, find all open paths (those where the start != end) and close them.
        openPathStarts = set([i for i in startsToEnds if startsToEnds[i] != i])
        closedPaths, pathStartToZValue = close_paths(paths, openPathStarts, startsToEnds, xCoords, halfDeltaX, yCoords, halfDeltaY, zValues)
        for i in closedPaths:
            paths[i] = closedPaths[i]  # Update the record of the paths with the closed path.
        for i in pathStartToZValue:
            # For all the newly closed paths, map their starting point to their interior Z value.
            pathsStartsToRegionZValues[i] = pathStartToZValue[i]

        # Determine the hierarchy of patches in order to determine where to make holes in the patches to prevent overlaps. This is necessary in order to
        # correctly plot smaller patches that are completely contained within a larger patch. With an alpha vale of 1, if a larger patch A is plotted after
        # a smaller patch B that is completely contained within A, then patch B will not be visible. If the alpha value is not 1, then the overlapping
        # patches will both be visible, but will combine their colors and be darker (due to the adding of overlapping alpha values) than the other patches.
        removeInternalPaths = calc_area_hierarchy([i for i in pathsStartsToRegionZValues], paths)

        # Fill in the area using the patches.
        for i in pathsStartsToRegionZValues:
            verts = paths[i]
            codes = [path.Path.MOVETO] + ([path.Path.LINETO] * (len(verts) - 1))
            for j in removeInternalPaths[i]:
                # For each path P that is completely contained within the current path, reverse P's vertices (so that it's clockwise) and add the
                # vertices to the current path's vertices. This will cause the smaller internal path to appear as a hole in the larger external patch.
                pathOfJ = paths[j]
                verts += pathOfJ[::-1]
                codes += [path.Path.MOVETO] + ([path.Path.LINETO] * (len(pathOfJ) - 1))
            boundary = path.Path(verts, codes)
            patch = patches.PathPatch(boundary, facecolor=colorMapping[pathsStartsToRegionZValues[i]], linewidth=0, edgecolor='none', alpha=fillAlpha, zorder=-1)
            axes.add_patch(patch)

    if outputLocation:
        plt.savefig(outputLocation, bbox_inches='tight', transparent=True)
    else:
        return currentFigure, axes


def calc_area_hierarchy(pathStarts, paths):
    """Determine the hierarchy of a set of paths.

    Determines the set of paths that are completely contained within each path. If there is a hierarchy such as A contains B which contains C, then this
    is marked only as A containing B and B containing C (A is not marked as containing C).

    :param pathStarts:  The vertices that correspond to the start of each path in paths.
    :type pathStarts:   list of (x, y) coordinate tuples
    :param paths:       The vertices that make up each of the paths.
    :type paths:        dict of lists of (x, y) coordinate tuples
    :returns :          A mapping from each path starting point to the (x, y) coordinates of the starting points of paths internal to it.
    :type :             dict

    """

    # Create the Path objects for all the enclosed areas.
    boundaries = dict([(i, path.Path(paths[i], [path.Path.MOVETO] + ([path.Path.LINETO] * (len(paths[i]) - 1)))) for i in pathStarts])

    # Determine the paths that are inside each path.
    nested = {}
    for i in pathStarts:
        boundary = boundaries[i]
        areasContained = [j for j in pathStarts if boundary.contains_path(boundaries[j])]
        nested[i] = areasContained

    # For each path that contains other paths inside it, determine which of the internal paths need to be marked as internal to it. For example, If there
    # is a hierarchy such as A contains B which contains C, then determine that although A contains both B and C, only B needs to be marked as internal
    # (C will be marked as internal to B).
    remove = {}
    for i in nested:
        # Find the fewest areas that cover all areas internal to i. For example, i may have two disjoint areas, A and B, inside it. In turn, A has 1 area
        # inside it and B has 3 areas inside it. Therefore, i has 5 areas inside it. We are looking to find the smallest number of areas inside i that can
        # cove all internal areas. In this case it would be A and B, as together they cover all internal areas.

        # First sort i's internal areas by the number of areas they have internal to them (in descending order).
        internalsSortedBySizeDecreasing = sorted(nested[i], key=lambda x : len(nested[x]), reverse=True)
        toRemove = set([])  # The minimal set of i's internal areas that can be used to cover all of i's internal areas.
        included = set([])  # The areas internal to i that have been covered by the areas in toRemove.
        for j in internalsSortedBySizeDecreasing:
            if not j in included:
                # If j has not been included in the set already, then it is a top level area inside i. Therefore, add it to toRemove.
                toRemove.add(j)
            # Indicate that j and all the areas inside it have been dealt with.
            included |= set(nested[j])

        remove[i] = toRemove

    return remove


def close_paths(paths, openStartingPoints, startsToEnds, xCoords, halfDeltaX, yCoords, halfDeltaY, zValues):
    """Close open paths.

    All these boundaries will have both start an ending points on an edge of the figure.

    :param paths:               The vertices that make up each of the paths.
    :type paths:                dict of lists of (x, y) coordinate tuples
    :param openStartingPoints:  The starting points of open paths.
    :type openStartingPoints:   set of (x, y) coordinate tuples
    :param startsToEnds:        A mapping from starting points of paths to the ending points of the paths
    :type startsToEnds:         dict
    :param xCoords:             The x coordinates where the Z values have been evaluated.
    :type xCoords:              2 dimensional numpy array
    :param halfDeltaX:          Half the distance between adjacent X coordinate values.
    :type halfDeltaX:           float
    :param yCoords:             The y coordinates where the Z values have been evaluated.
    :type yCoords:              2 dimensional numpy array
    :param halfDeltaY:          Half the distance between adjacent Y coordinate values.
    :type halfDeltaY:           float
    :param zValues:             The z value for each (x,y) pair.
    :type zValues:              2 dimensional numpy array
    :returns :                  Mapping from the starting positions to the now closed paths, the interior Z value for each closed path.
    :type :                     dict mapping (x, y) coordinate tuple to list of tuples, dict mapping (x, y) coordinate tuple to float

    """

    # Determine boundaries of the (X,Y) grid.
    xMin = xCoords.min()
    xMax = xCoords.max()
    deltaX = abs(np.ediff1d([xCoords[0,0], xCoords[0,1]]))[0]  # The distance between adjacent X coordinate values.
    yMin = yCoords.min()
    yMax = yCoords.max()
    deltaY = abs(np.ediff1d([yCoords[0,0], yCoords[1,0]]))[0]  # The distance between adjacent Y coordinate values.

    # Setup the return values.
    newPaths = {}  # Altered paths that have been closed.
    pathStartToZValue = {}  # Mappings from the starting points of paths to the Z value of the interior of the path.

    # First close paths that start and end on the same figure edge. These are paths that just form loops on one edge of te figure.
    closedStartPoints = set([])
    for i in openStartingPoints:
        currentPath = [j for j in paths[i]]
        startXCoord = i[0]
        startYCoord = i[1]
        endOfPath = currentPath[-1]
        endXCoord = endOfPath[0]
        endYCoord = endOfPath[1]
        if (startXCoord == endXCoord) or (startYCoord == endYCoord):
            # The path starts and ends on the same edge.
            currentPath.append(i)
            pathIsCCW, areaZValue = test_clockwise(currentPath, xCoords, halfDeltaX, yCoords, halfDeltaY, zValues)
            if pathIsCCW:
                # If the path is counter-clockwise, then record the path and its associated Z value.
                pathStartToZValue[i] = areaZValue
                newPaths[i] = currentPath
                closedStartPoints.add(i)
    openStartingPoints -= closedStartPoints

    # Next close paths that start and end on different edges of the figure.
    while openStartingPoints:
        currentStartingPoint = openStartingPoints.pop()
        currentPath = [i for i in paths[currentStartingPoint]]

        # Travel around the edges of the figure in a CCW motion to close the path. The current movement of the figure is determined in both the X and Y
        # directions. There are four possible general locations for the current position being examined: A) left edge, B) bottom, C) right, or D) top.
        # The changes in the current position that should take place for each case are:
        # A) the X value of the current position should not change and the Y should decrease.
        # B) the X value of the current position should increase and the Y should not change.
        # C) the X value of the current position should not change and the Y should increase.
        # D) the X value of the current position should decrease and the Y should not change.
        # Determine the current position and the changes in the X and Y values that are needed.
        currentPosition = currentPath[-1]
        currentXMovement = 0 if (np.allclose([currentPosition[0]], [xMax]) or np.allclose([currentPosition[0]], [xMin])) else (deltaX if np.allclose([currentPosition[1]], [yMin]) else -deltaX)
        currentYMovement = 0 if (np.allclose([currentPosition[1]], [yMax]) or np.allclose([currentPosition[1]], [yMin])) else (deltaY if np.allclose([currentPosition[0]], [xMax]) else -deltaY)
        while not np.allclose([currentPosition], [currentStartingPoint]):
            # Loop until the starting position of the path is reached.

            # Update the current position being examined.
            currentPosition = (currentPosition[0] + currentXMovement, currentPosition[1] + currentYMovement)

            # If the current position is the start of another path, then append the paths together as they must be part of the same area boundary.
            # As this may take the current position onto another edge of the figure, update the X and Y movements.
            otherPathStart = [i for i in openStartingPoints if np.allclose([currentPosition], [i])]
            if otherPathStart:
                currentPosition = otherPathStart[0]
                openStartingPoints -= set([currentPosition])
                currentPath.extend(paths[currentPosition])
                currentPosition = paths[currentPosition][-1]
                currentXMovement = 0 if (np.allclose([currentPosition[0]], [xMax]) or np.allclose([currentPosition[0]], [xMin])) else (deltaX if np.allclose([currentPosition[1]], [yMin]) else -deltaX)
                currentYMovement = 0 if (np.allclose([currentPosition[1]], [yMax]) or np.allclose([currentPosition[1]], [yMin])) else (deltaY if np.allclose([currentPosition[0]], [xMax]) else -deltaY)

            # Determine if the currentXMovement and currentYMovement need updating.
            if np.allclose([currentPosition[0]], [xMax + halfDeltaX]):
                # Gone past the rightmost edge of the figure while travelling along the bottom edge of the figure.
                currentPath.append((xMax, yMin))  # Add the bottom right corner to the path
                currentPosition = (xMax, yMin + halfDeltaY)  # Go up a half step from the bottom right corner.
                currentPath.append(currentPosition)  # Add the new point.
                # Now travelling upwards along the rightmost edge.
                currentXMovement = 0
                currentYMovement = deltaY
            elif np.allclose([currentPosition[0]], [xMin - halfDeltaX]):
                # Gone past the leftmost edge of the figure while travelling along the top edge of the figure.
                currentPath.append((xMin, yMax))  # Add the top left corner to the path
                currentPosition = (xMin, yMax - halfDeltaY)  # Go down a half step from the top left corner.
                currentPath.append(currentPosition)  # Add the new point.
                # Now travelling downwards along the leftmost edge.
                currentXMovement = 0
                currentYMovement = -deltaY
            elif np.allclose([currentPosition[1]], [yMax + halfDeltaY]):
                # Gone past the top edge of the figure while travelling along the rightmost edge of the figure.
                currentPath.append((xMax, yMax))  # Add the top right corner to the path
                currentPosition = (xMax - halfDeltaX, yMax)  # Go left a half step from the top right corner.
                currentPath.append(currentPosition)  # Add the new point.
                # Now travelling leftwards along the top edge.
                currentXMovement = -deltaX
                currentYMovement = 0
            elif np.allclose([currentPosition[1]], [yMin - halfDeltaY]):
                # Gone past the bottom edge of the figure while travelling along the leftmost edge of the figure.
                currentPath.append((xMin, yMin))  # Add the bottom left corner to the path
                currentPosition = (xMin + halfDeltaX, yMin)  # Go right a half step from the bottom left corner.
                currentPath.append(currentPosition)  # Add the new point.
                # Now travelling rightwards along the bottom edge.
                currentXMovement = deltaX
                currentYMovement = 0

        # Close the path now that the starting point has been reached.
        currentPath.append(currentStartingPoint)

        # Determine the Z value of the area enclosed by the path (this is the Z value of the point on the (X,Y) grid one half step CW from the starting point.
        if np.allclose([currentStartingPoint[0]], [xMin]):
            # Starting point is on the leftmost edge of the figure.
            pointToCheck = (currentStartingPoint[0], currentStartingPoint[1] + halfDeltaY)
        elif np.allclose([currentStartingPoint[0]], [xMax]):
            # Starting point is on the rightmost edge of the figure.
            pointToCheck = (currentStartingPoint[0], currentStartingPoint[1] - halfDeltaY)
        elif np.allclose([currentStartingPoint[1]], [yMin]):
            # Starting point is on the bottommost edge of the figure.
            pointToCheck = (currentStartingPoint[0] - halfDeltaX, currentStartingPoint[1])
        else:  # if np.allclose([currentStartingPoint[1]], [yMax])
            # Starting point is on the topmost edge of the figure.
            pointToCheck = (currentStartingPoint[0] + halfDeltaX, currentStartingPoint[1])
        indexY = np.where(xCoords.round(6) == np.around(pointToCheck[0], 6))[1][0]
        indexX = np.where(yCoords.round(6) == np.around(pointToCheck[1], 6))[0][0]
        areaZValue = zValues[indexX, indexY]

        # Record the path and Z value information for the starting point.
        pathStartToZValue[currentStartingPoint] = areaZValue
        newPaths[currentStartingPoint] = currentPath

    return newPaths, pathStartToZValue


def test_clockwise(pathVertices, xCoords, halfDeltaX, yCoords, halfDeltaY, zValues):
    """Test whether a path is counter-clockwise.

    Can be used with open paths, but is not guaranteed to return the correct result.

    :param pathVertices:    The vertices that make up the path.
    :type pathVertices:     list of (x, y) coordinate tuples
    :param xCoords:         The x coordinates where the Z values have been evaluated.
    :type xCoords:          2 dimensional numpy array
    :param halfDeltaX:      Half the distance between adjacent X coordinate values.
    :type halfDeltaX:       float
    :param yCoords:         The y coordinates where the Z values have been evaluated.
    :type yCoords:          2 dimensional numpy array
    :param halfDeltaY:      Half the distance between adjacent Y coordinate values.
    :type halfDeltaY:       float
    :param zValues:         The z value for each (x,y) pair.
    :type zValues:          2 dimensional numpy array
    :returns :              Whether the path s counter-clockwise, and the Z value of the interior of the path.
    :type :                 boolean, float

    """

    # Create the path object from the vertices.
    codes = [path.Path.MOVETO] + ([path.Path.LINETO] * (len(pathVertices) - 1))
    boundary = path.Path(pathVertices, codes)

    # Determine the endpoints of the first line segment of the path.
    firstSegmentStartX = pathVertices[0][0]
    firstSegmentStartY = pathVertices[0][1]
    firstSegmentEndX = pathVertices[1][0]
    firstSegmentEndY = pathVertices[1][1]

    # Determine the direction that the first line segment goes, and where is starts on the (X, Y) gridlines.
    # If the (X, Y) grid contains all x values in [0, 1] and all Y values in [0, 1], then halfDeltaX and halfDeltaY would be 0.5. Therefore,
    # the grid would be like:
    # (0,0)    (1,0)
    #
    # (0,1)    (1,1)
    # and the first line segment could start at A) (0, 0.5), B) (0.5, 1), C) (1, 0.5), or D) (0.5, 0).
    # A and C are on the X gridlines, while B and D are on the Y gridlines.
    verticalLine = firstSegmentStartX == firstSegmentEndX   # The fist line segment is vertical.
    horzontalLine = firstSegmentStartY == firstSegmentEndY  # The fist line segment is horizontal.
    up = firstSegmentStartY < firstSegmentEndY              # The fist line segment goes up (not necessarily vertical).
    down = firstSegmentStartY > firstSegmentEndY            # The fist line segment goes down (not necessarily vertical).
    left = firstSegmentStartX > firstSegmentEndX            # The fist line segment goes left (not necessarily horizontal).
    right = firstSegmentStartX < firstSegmentEndX           # The fist line segment goes right (not necessarily horizontal).
    onXGrid = np.where(xCoords == firstSegmentStartX)[0].size > 0  # True if the first line segment starts on a line on the X grid.
    onYGrid = np.where(yCoords == firstSegmentStartY)[0].size > 0  # True if the first line segment starts on a line on the Y grid.

    # A path will only be counter-clockwise if a specific point is in the interior of the path. This specific point depends on the direction that
    # the first line segment of the path goes, and is based on the splitting of the meshgrid into individual squares that have been evaluated separately.
    pointToCheck = None  # The point that should be checked to determine the Z value of the interior of the path.
    CCW = True  # Whether the path is counter-clockwise.
    if verticalLine and up:
        # If the first line segment goes vertically up, then it divides the square into a left and right. The two points that make up the left corners
        # of the square must be inside the path if it is to be counter-clockwise.
        pointToCheck = (firstSegmentStartX - halfDeltaX, firstSegmentStartY)  # The bottom left corner of the square.
    elif verticalLine and down:
        # If the first line segment goes vertically down, then it divides the square into a left and right. The two points that make up the right corners
        # of the square must be inside the path if it is to be counter-clockwise.
        pointToCheck = (firstSegmentStartX + halfDeltaX, firstSegmentStartY)  # The top right corner of the square.
    elif horzontalLine and left:
        # If the first line segment goes horizontal and left, then it divides the square into a top and bottom. The two points that make up the bottom corners
        # of the square must be inside the path if it is to be counter-clockwise.
        pointToCheck = (firstSegmentStartX, firstSegmentStartY - halfDeltaY)  # The bottom right corner of the square.
    elif horzontalLine and right:
        # If the first line segment goes horizontal and right, then it divides the square into a top and bottom. The two points that make up the top corners
        # of the square must be inside the path if it is to be counter-clockwise.
        pointToCheck = (firstSegmentStartX, firstSegmentStartY + halfDeltaY)  # The top left corner of the square.
    elif onYGrid and up:
        # If the first line segment starts on the Y grid lines and goes upwards, then it begins in middle of the bottom edge of the square, and goes either
        # up to the middle of the left edge or up to the middle of the right edge. The square therefore looks as follows:
        # X  X
        # \  /
        # P\/X
        # In order for the path to be counter-clockwise, point P must be on the interior of the path.
        pointToCheck = (firstSegmentStartX - halfDeltaX, firstSegmentStartY)  # Point P on the square.
    elif onYGrid and down:
        # If the first line segment starts on the Y grid lines and goes down, then it begins in middle of the top edge of the square, and goes either
        # down to the middle of the left edge or down to the middle of the right edge. The square therefore looks as follows:
        # X/\P
        # /  \
        # X  X
        # In order for the path to be counter-clockwise, point P must be on the interior of the path.
        pointToCheck = (firstSegmentStartX + halfDeltaX, firstSegmentStartY)  # Point P on the square.
    elif right:  # and onXGrid
        # If the first line segment starts on the X grid lines and goes right, then it begins in middle of the left edge of the square, and goes either
        # down to the middle of the bottom edge or up to the middle of the top edge. The square therefore looks as follows:
        # P/  X
        # /
        # \
        # X\  X
        # In order for the path to be counter-clockwise, point P must be on the interior of the path.
        pointToCheck = (firstSegmentStartX, firstSegmentStartY + halfDeltaY)  # Point P on the square.
    else:  # left and onXGrid
        # If the first line segment starts on the X grid lines and goes left, then it begins in middle of the right edge of the square, and goes either
        # down to the middle of the bottom edge or up to the middle of the top edge. The square therefore looks as follows:
        # X  \X
        #     \
        #     /
        # X  /P
        # In order for the path to be counter-clockwise, point P must be on the interior of the path.
        pointToCheck = (firstSegmentStartX, firstSegmentStartY - halfDeltaY)  # Point P on the square.

    # Inflate the path slightly when checking whether pointToCheck is inside the path as pointToCheck might be on the edge of the path (only if pointToCheck
    # is on the maximum value of the X or Y gridlines and the path therefore goes along the edge of the figure), as any point directly on the boundary of a
    # path does not count as inside the path. The inflation amount is small enough to not allow any points on the meshgrid that should not be inside the path
    # to be counted as inside it.
    CCW = boundary.contains_point(pointToCheck, radius=min(halfDeltaX, halfDeltaY) / 2)

    if not CCW:
        # The path is not counter-clockwise.
        return False, 'none'

    # Get the Z value of the interior of a counter-clockwise path. This is the Z value of the point on the (X, Y) at pointToCheck.
    # Rounding is used in order to accurately compare floating point numbers.
    indexY = np.where(xCoords.round(6) == np.around(pointToCheck[0], 6))[1][0]
    indexX = np.where(yCoords.round(6) == np.around(pointToCheck[1], 6))[0][0]
    return True, zValues[indexX, indexY]