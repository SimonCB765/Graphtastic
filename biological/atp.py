import matplotlib.patches as patches
import numpy as np


def main(startX, startY, baseRiboseRadius=0.2, basePhosphateRadius=0.1, scale=1.0, numberOfPhosphates=3):
    """Generate the patches needed to draw an ATP/ADP.

    :param startX:                  The X coordinate of the center of the ribose.
    :type startX:                   float
    :param startY:                  The Y coordinate of the center of the ribose
    :type startY:                   float
    :param baseRiboseRadius:        The base radius of the ribose that will be scaled appropriately.
    :type baseRiboseRadius:         float
    :param basePhosphateRadius:     The base radius of each phosphate that will be scaled appropriately.
    :type basePhosphateRadius:      float
    :param scale:                   How to scale the base ATP/ADP.
    :type scale:                    float
    :param numberOfPhosphates:      The number of phosphates to attach to the ribose.
    :type numberOfPhosphates:       int

    """

    # Generate the ribose. The default orientation for a pentagon is for vertex 1 to be directly above the center of the pentagon. Numbering the
    # vertices in a clockwise manner, the angle between vertex 2, the center and the horizontal is set to 72 degrees.
    riboseRadius = scale * baseRiboseRadius
    angleOfRotation = (72 * np.pi) / 180  # 72 degrees
    vertex2CenterHorizontalAngle = (np.pi / 2) - angleOfRotation
    ribose = patches.RegularPolygon((startX, startY), numVertices=5, radius=riboseRadius, orientation=-vertex2CenterHorizontalAngle)

    # Generate the phosphates.
    phosphates = []
    phosphateRadius = scale * basePhosphateRadius
    for i in range(numberOfPhosphates):
        phosphates.append(patches.Circle((startX + riboseRadius + ((1 + (2 * i)) * phosphateRadius), startY), phosphateRadius))

    return ribose, phosphates