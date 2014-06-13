import matplotlib.patches as patches
import matplotlib.path as mpath
import matplotlib.pyplot as plt

import sys

# Import the patch and path generation scripts.
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
import ionchannel
import membrane


def main(outputLocation):
    """Draw and ion channel embedded in a membrane.

    :param outputLocation:  The location to save the image.
    :type outputLocation:   string

    """

    # Setup the figure.
    minX = -1.0
    maxX = 3.0
    yMin = -0.5
    yMax = 1.75
    fig, ax = plt.subplots(subplot_kw={'aspect': 1.0, 'xlim': [minX, maxX], 'ylim': [yMin, yMax]})
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # Define the scaling for the ion channel.
    gateScale = 1

    #---------------------------------------#
    #     Draw the inactive ion channel     #
    #---------------------------------------#
    inactiveStartingXCoord = minX + 1
    inactiveLeftSubunitVertices, inactiveRightSubunitVertices, inactiveMovements = ionchannel.main(inactiveStartingXCoord, active=False, scale=gateScale)
    pathLeftInactiveSubunit = mpath.Path(inactiveLeftSubunitVertices, inactiveMovements)
    patchLeftInactiveSubunit = patches.PathPatch(pathLeftInactiveSubunit, facecolor='orange', lw=2)
    patchLeftInactiveSubunit.set_facecolor('0.8')
    ax.add_patch(patchLeftInactiveSubunit)
    pathRightInactiveSubunit = mpath.Path(inactiveRightSubunitVertices, inactiveMovements)
    patchRightInactiveSubunit = patches.PathPatch(pathRightInactiveSubunit, facecolor='orange', lw=2)
    patchRightInactiveSubunit.set_facecolor('0.8')
    ax.add_patch(patchRightInactiveSubunit)
    minInactiveLeftXCoord = min([i[0] for i in inactiveLeftSubunitVertices])
    maxInactiveRightXCoord = max([i[0] for i in inactiveRightSubunitVertices])
    ax.text(((minInactiveLeftXCoord + maxInactiveRightXCoord) / 2), -0.35, 'Closed', size=15, color='black', horizontalalignment='center', verticalalignment='center')

    #-------------------------------------#
    #     Draw the active ion channel     #
    #-------------------------------------#
    activeStartingXCoord = inactiveStartingXCoord + 1.5
    activeLeftSubunitVertices, activeRightSubunitVertices, activeMovements = ionchannel.main(activeStartingXCoord, active=True, scale=gateScale)
    pathLeftActiveSubunit = mpath.Path(activeLeftSubunitVertices, activeMovements)
    patchLeftActiveSubunit = patches.PathPatch(pathLeftActiveSubunit, facecolor='orange', lw=2)
    patchLeftActiveSubunit.set_facecolor('0.8')
    ax.add_patch(patchLeftActiveSubunit)
    pathRightActiveSubunit = mpath.Path(activeRightSubunitVertices, activeMovements)
    patchRightActiveSubunit = patches.PathPatch(pathRightActiveSubunit, facecolor='orange', lw=2)
    patchRightActiveSubunit.set_facecolor('0.8')
    ax.add_patch(patchRightActiveSubunit)
    minActiveLeftXCoord = min([i[0] for i in activeLeftSubunitVertices])
    maxActiveRightXCoord = max([i[0] for i in activeRightSubunitVertices])
    ax.text(((minActiveLeftXCoord + maxActiveRightXCoord) / 2), -0.35, 'Open', size=15, color='black', horizontalalignment='center', verticalalignment='center')

    #--------------------------------------#
    #            Draw some ions            #
    #--------------------------------------#
    ionWidth = 0.05
    ionInactiveCentersX = [0.10, 0.2, 0.2, 0.21, 0.23, 0.30, 0.55, 0.5, -0.04, 0.04, 0.4, 0.31, -0.18]
    ionInactiveCentersY = [1.15, 1.0, 1.25, 0.43, 0.6, 1.08, 1.15, 1.0, 1.15, 1.28, 1.27, 1.39, 1.07]
    ionsInactive = [patches.Circle((i, j), ionWidth / 2) for i, j in zip(ionInactiveCentersX, ionInactiveCentersY)]
    ionActiveCentersX = [1.41, 1.73, 1.7, 1.75, 1.58, 2.0, 1.82, 1.73, 1.64, 1.87, 1.65, 1.53, 1.84]
    ionActiveCentersY = [-0.08, 0.53, 0.34, -0.02, -0.17, -0.15, -0.22, 0.89, 1.08, 1.19, 1.27, 1.2, 1.32]
    ionsActive = [patches.Circle((i, j), ionWidth / 2) for i, j in zip(ionActiveCentersX, ionActiveCentersY)]
    for i in ionsInactive:
        i.set_facecolor('0.25')
        i.set_linewidth(0)
        ax.add_patch(i)
    for i in ionsActive:
        i.set_facecolor('0.25')
        i.set_linewidth(0)
        ax.add_patch(i)

    #--------------------------------------------------------------------------#
    #    Draw the activation arrow, selectivity filter and gate highlighting    #
    #--------------------------------------------------------------------------#
    # Add the activation arrow.
    arrowStartX = maxInactiveRightXCoord
    arrowEndX = minActiveLeftXCoord
    arrowY = 1.5
    ax.arrow(arrowStartX, arrowY, arrowEndX - arrowStartX, 0, head_width=0.1, head_length=0.1, fc='k', ec='k')
    ax.text(arrowStartX + ((arrowEndX - arrowStartX) / 2), arrowY + 0.1, 'Gating', size=12, color='black', horizontalalignment='center',
        verticalalignment='center')
    ax.text(arrowStartX + ((arrowEndX - arrowStartX) / 2), arrowY - 0.1, 'Stimulus', size=12, color='black', horizontalalignment='center',
        verticalalignment='center')

    # Circle the selectivity filter and gate.
    inactiveMidPointXCoord = ((minInactiveLeftXCoord + maxInactiveRightXCoord) / 2)
    highlightRadius = 0.1
    selectivityFilterY = 0.81 * gateScale
    selectivityFilterHighlight = patches.Circle((inactiveMidPointXCoord, selectivityFilterY), highlightRadius)
    selectivityFilterHighlight.set_facecolor('none')
    selectivityFilterHighlight.set_linewidth(2)
    ax.add_patch(selectivityFilterHighlight)
    ax.text(inactiveMidPointXCoord - (highlightRadius * 7 / 5), selectivityFilterY, 'S', size=14, color='black', horizontalalignment='right', verticalalignment='center')
    gateY = 0.13 * gateScale
    gateHighlight = patches.Circle((inactiveMidPointXCoord, gateY), highlightRadius)
    gateHighlight.set_facecolor('none')
    gateHighlight.set_linewidth(2)
    ax.add_patch(gateHighlight)
    ax.text(inactiveMidPointXCoord - (highlightRadius * 7 / 5), gateY, 'G', size=14, color='black', horizontalalignment='right', verticalalignment='center')

    #-------------------------------------#
    #          Draw the membrane          #
    #-------------------------------------#
    startPoints = [minX - 1, maxInactiveRightXCoord - 0.03, maxActiveRightXCoord - 0.03]
    stopPoints = [0.1, minActiveLeftXCoord + 0.1, maxX + 1]
    for start, stop in zip(startPoints, stopPoints):
        heads, tailVerts, tailMoves = membrane.main(topLayerY=0.9, bottomLayerY=0.1, startPointLeft=start, stopPointRight=stop, headRadius=0.06)
        for i in heads:
            i.set_zorder(0)
            i.set_facecolor('white')
            ax.add_patch(i)
        for i, j in zip(tailVerts, tailMoves):
            path = mpath.Path(i, j)
            patch = patches.PathPatch(path, lw=2)
            patch.set_zorder(-1)
            ax.add_patch(patch)

    plt.savefig(outputLocation, bbox_inches='tight', transparent=True)


if __name__ == '__main__':
    main(sys.argv[1])