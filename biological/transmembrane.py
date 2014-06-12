import matplotlib.patches as patches
import matplotlib.path as mpath
import matplotlib.transforms as transforms
import numpy as np


def main(numTransMemRegions, startingXCoordinate, yCoordinate, axes=None, rotationAngle=0, transRegionWidth=0.3, transRegionHeight=1, gapBetweenRegions=-0.025):
    """Generate the patches needed to draw transmembrane regions.

    :param numTransMemRegions:      The number of transmembrane regions to generate.
    :type numTransMemRegions:       int
    :param startingXCoordinate:     The X coordinate of the lower left corner of the first transmembrane region.
    :type startingXCoordinate:      float
    :param yCoordinate:             The Y coordinate of the lower left corner of each transmembrane region.
    :type yCoordinate:              float
    :param axes:                    The matplotlib axes on which the transmembrane regions will be placed (only needed if rotation is being used).
    :type axes:                     matplotlib axes instance
    :param rotationAngle:           The angle (in degrees) by which the regions should be rotated.
    :type rotationAngle:            float
    :param transRegionWidth:        The width of the transmembrane regions.
    :type transRegionWidth:         float
    :param transRegionHeight:       The height of the transmembrane regions.
    :type transRegionHeight:        float
    :param gapBetweenRegions:       The space between each transmembrane region. A negative value for this will cause the regions to overlap.
    :type gapBetweenRegions:        float
    :returns :                      The transmembrane regions.
    :type :                         list of matplotlib.patches

    """

    rotationAngle = (rotationAngle * np.pi) / 180  # Convert rotation angle from degrees to radians.
    transMemPathces = []
    for i in range(numTransMemRegions):
        patch = patches.Rectangle(xy=[startingXCoordinate + ((transRegionWidth + gapBetweenRegions) * i), yCoordinate], width=transRegionWidth, height=transRegionHeight)
        if rotationAngle != 0:
            transformation = transforms.Affine2D().rotate_around(startingXCoordinate + ((transRegionWidth + gapBetweenRegions) * i), yCoordinate, rotationAngle) + axes.transData
            patch.set_transform(transformation)
        transMemPathces.append(patch)

    return transMemPathces


def gpcr_loops(numTransMemRegions, startingXCoordinate, yCoordinate, rotationAngle=0, transRegionWidth=0.3, transRegionHeight=1, gapBetweenRegions=-0.025):
    """Generate the loops connecting GPCR transmembrane regions.

    :param numTransMemRegions:      The number of transmembrane regions that were generated.
    :type numTransMemRegions:       int
    :param startingXCoordinate:     The X coordinate of the lower left corner of the first transmembrane region.
    :type startingXCoordinate:      float
    :param yCoordinate:             The Y coordinate of the lower left corner of each transmembrane region.
    :type yCoordinate:              float
    :param rotationAngle:           The angle (in degrees) by which the regions should be rotated.
    :type rotationAngle:            float
    :param transRegionWidth:        The width of the transmembrane regions.
    :type transRegionWidth:         float
    :param transRegionHeight:       The height of the transmembrane regions.
    :type transRegionHeight:        float
    :param gapBetweenRegions:       The space between each transmembrane region. A negative value for this will cause the regions to overlap.
    :type gapBetweenRegions:        float
    :returns :                      The vertices needed to draw the loops and strands for a GPCR.
    :type :                         list of lists of tuples, list of lists of tuples, list of tuples, list of tuples

    """

    rotationAngle = (rotationAngle * np.pi) / 180  # Convert the rotation angle from degrees to radians.

    #--------------------------------#
    #          Bottom Loops          #
    #--------------------------------#
    # Determine the X and Y coordinates of the bottom loops. The X coordinate of a loop i is the midpoint between the X coordinates of the bottom left corners
    # of the ith and i+1th transmembrane regions starting with the leftmost region. The Y coordinate is the point on the bottom edge of a region directly
    # above or below the X coordinate. It will only be equal to yCoordinate when there is no rotation of the regions.
    bottomLeftCornerTransMemRegionXCoords = [startingXCoordinate + ((transRegionWidth + gapBetweenRegions) * i) for i in range(numTransMemRegions)]
    bottomLoopXCoords = [i + ((transRegionWidth / 2) * np.cos(rotationAngle)) for i in bottomLeftCornerTransMemRegionXCoords]
    bottomLoopYCoord = yCoordinate + ((transRegionWidth / 2) * np.sin(rotationAngle))

    # Create the loop vertices.
    bottomVertices = []
    for i, j in zip(bottomLoopXCoords[0:-1:2], bottomLoopXCoords[1:-1:2]):
        loopVerts = []
        loopVerts.append((i, bottomLoopYCoord))
        loopVerts.append(((i + (1 / 7) * (transRegionWidth + gapBetweenRegions)), bottomLoopYCoord - (0.2 * transRegionHeight)))
        loopVerts.append(((i + (4 / 7) * (transRegionWidth + gapBetweenRegions)), bottomLoopYCoord - (0.2 * transRegionHeight)))
        loopVerts.append((j, bottomLoopYCoord))
        bottomVertices.append(loopVerts)

    #-------------------------------#
    #           Top Loops           #
    #-------------------------------#
    # Determine the X and Y coordinates of the top loops. The X coordinate of a loop i is the midpoint between the X coordinates of the top left corners
    # of the ith and i+1th transmembrane regions starting with the leftmost region. The Y coordinate is the point on the top edge of a region directly
    # above or below the X coordinate. It will only be equal to (yCoordinate + transRegionHeight) when there is no rotation of the regions.
    complementOfRotation = (np.pi / 2) - rotationAngle
    topLoopXCoords = [i - (transRegionHeight * np.cos(complementOfRotation)) for i in bottomLoopXCoords]
    topLoopYCoord = bottomLoopYCoord + (transRegionHeight * np.sin(complementOfRotation))

    # Create the loop vertices.
    topVertices = []
    for i, j in zip(topLoopXCoords[1::2], topLoopXCoords[2::2]):
        loopVerts = []
        loopVerts.append((i, topLoopYCoord))
        loopVerts.append(((i + (1 / 7) * (transRegionWidth + gapBetweenRegions)), topLoopYCoord + (0.2 * transRegionHeight)))
        loopVerts.append(((i + (4 / 7) * (transRegionWidth + gapBetweenRegions)), topLoopYCoord + (0.2 * transRegionHeight)))
        loopVerts.append((j, topLoopYCoord))
        topVertices.append(loopVerts)

    #----------------------------------------#
    #        N and C terminal Strands        #
    #----------------------------------------#
    # Create the N terminal strand.
    startXCoord = topLoopXCoords[0]
    startYCoord = topLoopYCoord
    vertsNStrand = [(startXCoord, startYCoord),
                    (startXCoord, startYCoord + transRegionWidth),
                    (startXCoord - (transRegionWidth * 2), startYCoord),
                    (startXCoord - (transRegionWidth * 2), startYCoord + (transRegionWidth * 2)),
                   ]

    # Create the C terminal strand.
    startXCoord = bottomLoopXCoords[-1]
    startYCoord = bottomLoopYCoord
    vertsCStrand = [(startXCoord, startYCoord),
                    (startXCoord, startYCoord - transRegionWidth),
                    (startXCoord + (transRegionWidth * 2), startYCoord),
                    (startXCoord, startYCoord - (transRegionWidth * 2)),
                   ]

    return bottomVertices, topVertices, vertsNStrand, vertsCStrand