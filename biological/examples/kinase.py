import matplotlib.patches as patches
import matplotlib.path as mpath
import matplotlib.pyplot as plt

import sys

# Import the patch and path generation scripts.
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
import atp
import kinase


def main(outputLocation):
    """Draw a kinase phosphorylating its substrate.

    :param outputLocation:  The location to save the image.
    :type outputLocation:   string

    """

    # Setup the figure.
    xMin = 0.0
    xMax = 12.0
    yMin = 0.0
    yMax = 8.0
    fig, ax = plt.subplots(subplot_kw={'aspect': 1.0, 'xlim': [xMin, xMax], 'ylim': [yMin, yMax]})
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # Setup the scale of the images.
    scaleKinase = 1.0
    scaleATPUnbound = 1.0
    scaleATPBound = 0.75
    baseRiboseRadius = 0.2

    # Draw the kinases.
    kinases = []
    kinaseXCoords = [xMin + 1.5, xMin + 5.0, xMin + 8.5, xMin + 8.5, xMin + 5.0]
    kinaseYCoords = [yMax - 1.0, yMax - 1.0, yMax - 1.0, yMax - 5.5, yMax - 5.5]
    kinaseLabels = [True, False, False, False, False]
    for i in zip(kinaseXCoords, kinaseYCoords, kinaseLabels):
        kinaseValues = kinase.main(i[0], i[1], scale=scaleKinase)
        kinases.append(kinaseValues)
        nSubunitPatch, cSubunitPatch, hingePatch, substrateNLobeConnection, substrateCLobeConnection, labels = kinaseValues
        nSubunitPatch.set_facecolor('0.9')
        ax.add_patch(nSubunitPatch)
        cSubunitPatch.set_facecolor('0.9')
        ax.add_patch(cSubunitPatch)
        ax.add_patch(hingePatch)
        if i[2]:
            for j in labels:
                plt.text(labels[j][0], labels[j][1], j, size=12 * scaleKinase, color='black', horizontalalignment='right', verticalalignment='center')

    # Draw the substrates.
    substrates = []
    differenceInLobeConnectionsX = kinases[0][3][0] - kinases[0][4][0]
    differenceInLobeConnectionsY = kinases[0][3][1] - kinases[0][4][1]
    unboundUnphosphedSubFirstVertex = (xMin + 6.5, yMax - 0.5)
    unboundUnphosphedSubSecondVertex = (unboundUnphosphedSubFirstVertex[0] - differenceInLobeConnectionsX, unboundUnphosphedSubFirstVertex[1] - differenceInLobeConnectionsY)
    unboundPhosphedSubFirstVertex = (xMin + 6.5, yMax - 5.0)
    unboundPhosphedSubSecondVertex = (unboundPhosphedSubFirstVertex[0] - differenceInLobeConnectionsX, unboundPhosphedSubFirstVertex[1] - differenceInLobeConnectionsY)
    substrateNLobeLocs = [unboundUnphosphedSubFirstVertex, kinases[2][3], kinases[3][3], unboundPhosphedSubFirstVertex]
    substrateCLobeLocs = [unboundUnphosphedSubSecondVertex, kinases[2][4], kinases[3][4], unboundPhosphedSubSecondVertex]
    substrateLabels = [True, False, False, False]
    for i in zip(substrateNLobeLocs, substrateCLobeLocs, substrateLabels):
        substrateValues = kinase.substrate(i[0], i[1], scale=scaleKinase)
        substrates.append(substrateValues)
        substrate, phosphateConnectionPoint, labelPos = substrateValues
        substrate.set_facecolor('0.7')
        ax.add_patch(substrate)
        if i[2]:
            plt.text(labelPos[0], labelPos[1], 'Substrate', size=12 * scaleKinase, color='black', horizontalalignment='right', verticalalignment='center')

    # Draw the ATPs and ADPs.
    ATPs = []
    atpXCoords = [xMin + 3.0, xMin + 5.0, xMin + 8.5, xMin + 8.5, xMin + 5.0, xMin + 3.0]
    atpYCoords = [yMax - 0.5, yMax - 1.5, yMax - 1.5, yMax - 6.0, yMax - 6.0, yMax - 5.0]
    atpLabels = [True, False, False, False, False, False]
    numberPhosphates = [3, 3, 3, 3, 3, 2]
    transferPhosph = [False, False, False, substrates[2][1], substrates[3][1], False]
    scale = [scaleATPUnbound, scaleATPBound, scaleATPBound, scaleATPBound, scaleATPBound, scaleATPUnbound]
    for i in zip(atpXCoords, atpYCoords, atpLabels, numberPhosphates, transferPhosph, scale):
        atpValues = atp.main(i[0], i[1], baseRiboseRadius=baseRiboseRadius, scale=i[5], numberOfPhosphates=i[3])
        ATPs.append(atpValues)
        ribose, phosphates = atpValues
        ribose.set_facecolor('0.3')
        ax.add_patch(ribose)
        for j in phosphates:
            j.set_facecolor('black')
            j.set_linewidth(0)
            ax.add_patch(j)
        if i[2]:
            plt.text(i[0] - (baseRiboseRadius * i[5]), i[1], 'ATP', size=12 * i[5], color='black', horizontalalignment='right', verticalalignment='center')
        if i[4]:
            phosphates[-1].center = i[4]  # Move a phosphate to the substrate.

    # Draw the arrows showing the flow of phosphorylation.
    unboundKinaseExitPoint = (kinaseXCoords[0] + 0.8, kinaseYCoords[0] - 0.65)
    unboundATPExitPoint = (atpXCoords[0], atpYCoords[0] - 0.35)
    kinaseTwoEntryPoint = (kinaseXCoords[1] - 0.6, kinaseYCoords[1] - 0.65)
    unboundSubstrateExitPoint = (unboundUnphosphedSubSecondVertex[0], unboundUnphosphedSubSecondVertex[1] - 0.15)
    kinaseTwoExitPoint = (kinaseXCoords[1] + 0.8, kinaseYCoords[1] - 0.65)
    kinaseThreeEntryPoint = (kinaseXCoords[2] - 0.6, kinaseYCoords[2] - 0.65)
    kinaseThreeExitPoint = (kinaseXCoords[2] + 0.2, kinaseYCoords[2] - 1.65)
    kinaseFourEntryPoint = (kinaseXCoords[3] + 0.2, kinaseYCoords[3] + 0.25)
    kinaseFourExitPoint = (kinaseXCoords[3] - 0.6, kinaseYCoords[3] - 0.65)
    unboundPhosphedSubstrate = (unboundPhosphedSubSecondVertex[0], unboundPhosphedSubSecondVertex[1] - 0.15)
    kinaseFiveEntryPoint = (kinaseXCoords[4] + 0.7, kinaseYCoords[4] - 0.65)
    kinaseFiveExitPoint = (kinaseXCoords[4] - 0.6, kinaseYCoords[4] - 0.65)
    unboundADPEntryPoint = (atpXCoords[-1], atpYCoords[-1] - 0.35)
    unboundKinaseEntryPoint = (kinaseXCoords[0] + 0.2, kinaseYCoords[0] - 1.55)
    ax.add_patch(patches.FancyArrowPatch(posA=unboundKinaseExitPoint, posB=kinaseTwoEntryPoint, shrinkA=0.0, shrinkB=0.0,
                                         arrowstyle='->,head_length=5.0,head_width=5.0'))
    ax.add_patch(patches.FancyArrowPatch(posA=unboundATPExitPoint, posB=kinaseTwoEntryPoint, shrinkA=0.0, shrinkB=0.0,
                                         arrowstyle='->,head_length=5.0,head_width=5.0', connectionstyle='angle3,angleA=270,angleB=0'))
    ax.add_patch(patches.FancyArrowPatch(posA=kinaseTwoExitPoint, posB=kinaseThreeEntryPoint, shrinkA=0.0, shrinkB=0.0,
                                         arrowstyle='->,head_length=5.0,head_width=5.0'))
    ax.add_patch(patches.FancyArrowPatch(posA=unboundSubstrateExitPoint, posB=kinaseThreeEntryPoint, shrinkA=0.0, shrinkB=0.0,
                                         arrowstyle='->,head_length=5.0,head_width=5.0', connectionstyle='angle3,angleA=270,angleB=0'))
    ax.add_patch(patches.FancyArrowPatch(posA=kinaseThreeExitPoint, posB=kinaseFourEntryPoint, shrinkA=0.0, shrinkB=0.0,
                                         arrowstyle='->,head_length=5.0,head_width=5.0'))
    ax.add_patch(patches.FancyArrowPatch(posA=kinaseFourExitPoint, posB=kinaseFiveEntryPoint, shrinkA=0.0, shrinkB=0.0,
                                         arrowstyle='->,head_length=5.0,head_width=5.0'))
    ax.add_patch(patches.FancyArrowPatch(posA=kinaseFourExitPoint, posB=unboundPhosphedSubstrate, shrinkA=0.0, shrinkB=0.0,
                                         arrowstyle='->,head_length=5.0,head_width=5.0', connectionstyle='angle3,angleA=0,angleB=270'))
    ax.add_patch(patches.FancyArrowPatch(posA=kinaseFiveExitPoint, posB=unboundKinaseEntryPoint, shrinkA=0.0, shrinkB=0.0,
                                         arrowstyle='->,head_length=5.0,head_width=5.0', connectionstyle='angle,angleA=0,angleB=270,rad=20'))
    ax.add_patch(patches.FancyArrowPatch(posA=kinaseFiveExitPoint, posB=unboundADPEntryPoint, shrinkA=0.0, shrinkB=0.0,
                                         arrowstyle='->,head_length=5.0,head_width=5.0', connectionstyle='angle3,angleA=0,angleB=270'))

    # Add the annotations on the arrows.
    ax.text(unboundKinaseExitPoint[0] + 0.3, unboundKinaseExitPoint[1] - 0.2, 'ATP\nbinding', size=12 * scaleKinase, color='black', horizontalalignment='left', verticalalignment='top')
    ax.text(kinaseTwoExitPoint[0] + 0.4, kinaseTwoExitPoint[1] - 0.2, 'Substrate\nbinding', size=12 * scaleKinase, color='black', horizontalalignment='left', verticalalignment='top')
    ax.text(kinaseThreeExitPoint[0] + 0.1, kinaseThreeExitPoint[1] - ((kinaseThreeExitPoint[1] - kinaseFourEntryPoint[1]) / 2), 'Phophoryl\ntransfer', size=12 * scaleKinase, color='black', horizontalalignment='left', verticalalignment='center')
    ax.text(kinaseTwoExitPoint[0] + 0.4, kinaseFourExitPoint[1] - 0.2, 'Substrate\nrelease', size=12 * scaleKinase, color='black', horizontalalignment='left', verticalalignment='top')
    ax.text(unboundKinaseExitPoint[0] + 0.3, kinaseFiveExitPoint[1] - 0.2, 'ADP\nrelease', size=12 * scaleKinase, color='black', horizontalalignment='left', verticalalignment='top')

    plt.savefig(outputLocation, bbox_inches='tight', transparent=True)


if __name__ == '__main__':
    main(sys.argv[1])