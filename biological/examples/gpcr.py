import matplotlib.path as mpath
import matplotlib.patches as patches
import matplotlib.pyplot as plt

import sys

# Import the patch and path generation scripts.
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
import gprotein
import membrane
import transmembrane


def main(outputLocation):
    """Draw a GPCR embedded in a membrane.

    :param outputLocation:  The location to save the image.
    :type outputLocation:   string

    """

    # Setup the figure.
    xMin = 0.25
    xMax = 9.25
    yMin = -1.0
    yMax = 2.0
    fig, ax = plt.subplots(subplot_kw={'aspect': 1.0, 'xlim': [xMin, xMax], 'ylim': [yMin, yMax]})
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # Define the transmembrane region properties.
    rotationAngle = 10
    numTransMemRegions = 7
    transRegionWidth = 0.3
    transRegionHeight = 1
    gapBetweenRegions = -0.025

    #-------------------------------------#
    #          Draw the membrane          #
    #-------------------------------------#
    startPoints = [xMin - 1]
    stopPoints = [xMax + 1]
    for start, stop in zip(startPoints, stopPoints):
        heads, tailVerts, tailMoves = membrane.main(topLayerY=0.9, bottomLayerY=0.1, startPointLeft=start, stopPointRight=stop, headRadius=0.06)
        for i in heads:
            i.set_zorder(-1)
            i.set_facecolor('white')
            ax.add_patch(i)
        for i, j in zip(tailVerts, tailMoves):
            path = mpath.Path(i, j)
            patch = patches.PathPatch(path, lw=2)
            patch.set_zorder(-2)
            ax.add_patch(patch)

    #-----------------------------------------------------------#
    #          Draw the GPCR with associated G protein          #
    #-----------------------------------------------------------#
    transmemStartXCoordAssociated = 1.0
    transmembraneYCoordinate = 0.0

    # Determine and draw the transmembrane regions.
    transMemPatchesAssociated = transmembrane.main(numTransMemRegions, transmemStartXCoordAssociated, transmembraneYCoordinate, ax, rotationAngle,
                                                   transRegionWidth, transRegionHeight, gapBetweenRegions)
    for i in transMemPatchesAssociated:
        i.set_facecolor('0.9')
        i.set_zorder(0)
        ax.add_patch(i)

    # Create the paths for all the loops and strands.
    movements = [mpath.Path.MOVETO,
                 mpath.Path.CURVE4,
                 mpath.Path.CURVE4,
                 mpath.Path.CURVE4
                ]

    # Determine vertices for the loops and strands.
    bottomLoopVertices, topLoopVertices, nStrandVertices, cStrandVertices = transmembrane.gpcr_loops(
            numTransMemRegions, transmemStartXCoordAssociated, transmembraneYCoordinate, rotationAngle,
            transRegionWidth, transRegionHeight, gapBetweenRegions
            )

    # Draw the bottom loops and C terminal strand.
    bottomLoopYCoord = bottomLoopVertices[0][0][1]
    for i, j in zip(bottomLoopVertices, [1, 2, 3]):
        path = mpath.Path(i, movements)
        patch = patches.PathPatch(path, facecolor='none', lw=2)
        ax.add_patch(patch)
        ax.text(i[0][0] + ((transRegionWidth + gapBetweenRegions) / 2), bottomLoopYCoord - (0.2 * transRegionHeight), 'IL' + str(j), size=(10 * transRegionHeight), color='black', horizontalalignment='center',
            verticalalignment='top')
    cTerminalPath = mpath.Path(cStrandVertices, movements)
    cTerminalPatch = patches.PathPatch(cTerminalPath, facecolor='none', lw=2)
    ax.add_patch(cTerminalPatch)
    ax.text(cStrandVertices[-1][0], cStrandVertices[-1][1] - (0.2 * transRegionHeight), 'C Terminus', size=(10 * transRegionHeight), color='black', horizontalalignment='right')

    # Draw the top loops and N terminal strand.
    topLoopYCoord = topLoopVertices[0][0][1]
    for i, j in zip(topLoopVertices, [1, 2, 3]):
        path = mpath.Path(i, movements)
        patch = patches.PathPatch(path, facecolor='none', lw=2)
        ax.add_patch(patch)
        ax.text(i[0][0] + ((transRegionWidth + gapBetweenRegions) / 2), topLoopYCoord + (0.4 * transRegionHeight), 'EL' + str(j), size=(10 * transRegionHeight), color='black', horizontalalignment='center',
            verticalalignment='top')
    nTerminalPath = mpath.Path(nStrandVertices, movements)
    nTerminalPatch = patches.PathPatch(nTerminalPath, facecolor='none', lw=2)
    ax.add_patch(nTerminalPatch)
    ax.text(nStrandVertices[-1][0], nStrandVertices[-1][1] + (0.1 * transRegionHeight), 'N Terminus', size=(10 * transRegionHeight), color='black')

    # Generate and draw the associated G protein.
    associatedGProteinXCoord = 3.4
    associatedGProteinYCoord = -0.2
    gAlphaAssociated, gBetaAssociated, gGammaAssociated, boundGDP, labelCoords = gprotein.main((associatedGProteinXCoord, associatedGProteinYCoord), label=True, scale=1.0, GTP=True)
    gAlphaAssociated.set_facecolor('0.75')
    ax.add_patch(gAlphaAssociated)
    gBetaAssociated.set_facecolor('0.6')
    ax.add_patch(gBetaAssociated)
    gGammaAssociated.set_facecolor('0.6')
    ax.add_patch(gGammaAssociated)
    boundGDP.set_facecolor('black')
    ax.add_patch(boundGDP)
    plt.text(labelCoords['alpha'][0], labelCoords['alpha'][1], r'$\alpha$', size=16, color='black', horizontalalignment='center', verticalalignment='center')
    plt.text(labelCoords['beta'][0], labelCoords['beta'][1], r'$\beta$', size=16, color='black', horizontalalignment='center', verticalalignment='center')
    plt.text(labelCoords['gamma'][0], labelCoords['gamma'][1], r'$\gamma$', size=16, color='black', horizontalalignment='center', verticalalignment='center')
    plt.text(labelCoords['GTP'][0], labelCoords['GTP'][1], 'GDP', size=12, color='black', horizontalalignment='center', verticalalignment='top')

    #--------------------------------------------------------------#
    #          Draw the GPCR with disassociated G protein          #
    #--------------------------------------------------------------#
    transmemStartXCoordDisassociated = 5.5
    transmembraneYCoordinate = 0.0

    # Determine and draw the transmembrane regions.
    transMemPatchesDisassociated = transmembrane.main(numTransMemRegions, transmemStartXCoordDisassociated, transmembraneYCoordinate, ax, rotationAngle,
                                                      transRegionWidth, transRegionHeight, gapBetweenRegions)
    for i in transMemPatchesDisassociated:
        i.set_facecolor('0.9')
        i.set_zorder(0)
        ax.add_patch(i)

    # Create the paths for all the loops and strands.
    movements = [mpath.Path.MOVETO,
                 mpath.Path.CURVE4,
                 mpath.Path.CURVE4,
                 mpath.Path.CURVE4
                ]

    # Determine vertices for the loops and strands.
    bottomLoopVertices, topLoopVertices, nStrandVertices, cStrandVertices = transmembrane.gpcr_loops(
            numTransMemRegions, transmemStartXCoordDisassociated, transmembraneYCoordinate, rotationAngle,
            transRegionWidth, transRegionHeight, gapBetweenRegions
            )

    # Draw the bottom loops and C terminal strand.
    bottomLoopYCoord = bottomLoopVertices[0][0][1]
    for i in bottomLoopVertices:
        path = mpath.Path(i, movements)
        patch = patches.PathPatch(path, facecolor='none', lw=2)
        ax.add_patch(patch)
    cTerminalPath = mpath.Path(cStrandVertices, movements)
    cTerminalPatch = patches.PathPatch(cTerminalPath, facecolor='none', lw=2)
    ax.add_patch(cTerminalPatch)

    # Draw the top loops and N terminal strand.
    topLoopYCoord = topLoopVertices[0][0][1]
    for i in topLoopVertices:
        path = mpath.Path(i, movements)
        patch = patches.PathPatch(path, facecolor='none', lw=2)
        ax.add_patch(patch)
    nTerminalPath = mpath.Path(nStrandVertices, movements)
    nTerminalPatch = patches.PathPatch(nTerminalPath, facecolor='none', lw=2)
    ax.add_patch(nTerminalPatch)

    # Generate and draw the disassociated G protein.
    disassociatedOffset = transmemStartXCoordDisassociated - transmemStartXCoordAssociated
    gAlphaDisassociated, gBetaDisassociated, gGammaDisassociated, boundGTP, labelCoords = gprotein.main((associatedGProteinXCoord + disassociatedOffset + transRegionWidth, associatedGProteinYCoord), label=True, scale=1.0, GTP=True, disassociate=True)
    gAlphaDisassociated.set_facecolor('0.75')
    ax.add_patch(gAlphaDisassociated)
    gBetaDisassociated.set_facecolor('0.6')
    ax.add_patch(gBetaDisassociated)
    gGammaDisassociated.set_facecolor('0.6')
    ax.add_patch(gGammaDisassociated)
    boundGTP.set_facecolor('black')
    ax.add_patch(boundGTP)
    plt.text(labelCoords['alpha'][0], labelCoords['alpha'][1], r'$\alpha$', size=16, color='black', horizontalalignment='center', verticalalignment='center')
    plt.text(labelCoords['beta'][0], labelCoords['beta'][1], r'$\beta$', size=16, color='black', horizontalalignment='center', verticalalignment='center')
    plt.text(labelCoords['gamma'][0], labelCoords['gamma'][1], r'$\gamma$', size=16, color='black', horizontalalignment='center', verticalalignment='center')
    plt.text(labelCoords['GTP'][0], labelCoords['GTP'][1], 'GTP', size=12, color='black', horizontalalignment='center', verticalalignment='top')

    #---------------------------------------------#
    #          Draw the activation arrow          #
    #---------------------------------------------#
    arrowStartX = transmemStartXCoordAssociated + ((transRegionWidth + gapBetweenRegions) * 8)
    arrowEndX = transmemStartXCoordDisassociated - ((transRegionWidth + gapBetweenRegions) * 4)
    arrowY = yMax - 0.5
    ax.arrow(arrowStartX, arrowY, arrowEndX - arrowStartX, 0, head_width=0.1, head_length=0.1, fc='k', ec='k')
    ax.text(arrowStartX + ((arrowEndX - arrowStartX) / 2), arrowY + 0.15, 'Ligand', size=10, color='black', horizontalalignment='center',
        verticalalignment='center')
    ax.text(arrowStartX + ((arrowEndX - arrowStartX) / 2), arrowY - 0.15, 'Activation', size=10, color='black', horizontalalignment='center',
        verticalalignment='center')

    plt.savefig(outputLocation, bbox_inches='tight', transparent=True)

if __name__ == '__main__':
    main(sys.argv[1])