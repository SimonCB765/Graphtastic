import matplotlib.patches as patches
import matplotlib.path as mpath


def main(startX, startY, scale=1.0):
    """Generate the objects needed to draw a kinase.

    :param startX:      The X coordinate of the first point on the kinase.
    :type startX:       float
    :param startY:      The Y coordinate of the first point on the kinase.
    :type startY:       float
    :param scale:       How much to scale the base kinase.
    :type scale:        float
    :returns :          The N subunit, C subunit, hinge region, connection coordinates for the substrate on the N subunit, connection coordinates
                        for the substrate on the C subunit and the labels for the kinase.
    :type :             matplotlib.patches object, matplotlib.patches object, matplotlib.patches object, list of two floats, list of two floats,
                        dictionary of tuples each containing two floats

    """

    # The coordinates for drawing the kinase's N subunit.
    nSubunitV1 = (startX, startY)
    nSubunitCP1ForV1ToV2 = (startX + (0.8 * scale), startY + (0.2 * scale))
    nSubunitCP2ForV1ToV2 = (startX + (0.8 * scale), startY - (0.8 * scale))
    nSubunitV2 = (startX - (0.25 * scale), startY - (0.65 * scale))
    nSubunitCP1ForV2ToV1 = (startX - (0.5 * scale), startY - (0.6 * scale))
    nSubunitCP2ForV2ToV1 = (startX - (0.5 * scale), startY + (-0.1 * scale))

    # The coordinates for drawing the kinase's C subunit.
    cSubunitV1 = (startX - (0.15 * scale), startY - (0.6 * scale))
    cSubunitCP1ForV1ToV2 = (startX + (1.0 * scale), startY - (0.35 * scale))
    cSubunitCP2ForV1ToV2 = (startX + (1.0 * scale), startY - (1.35 * scale))
    cSubunitV2 = (startX + (0.25 * scale), startY - (1.45 * scale))
    cSubunitCP1ForV2ToV1 = (startX - (0.45 * scale), startY - (1.55 * scale))
    cSubunitCP2ForV2ToV1 = (startX - (0.45 * scale), startY - (0.65 * scale))

    # Calculate the coordinates where the hinge region connects to the N and C subunits.
    timePointHingeStart = 0.25
    hingeStartX, hingeStartY = calc_cubic_Bezier_pos(nSubunitV2, nSubunitCP1ForV2ToV1, nSubunitCP2ForV2ToV1, nSubunitV1, timePointHingeStart)
    timePointHingeEnd = 0.65
    hingeStopX, hingeStopY = calc_cubic_Bezier_pos(cSubunitV2, cSubunitCP1ForV2ToV1, cSubunitCP2ForV2ToV1, cSubunitV1, timePointHingeEnd)

    # Calculate the coordinates where the substrate connects to the N and C subunits.
    timePointSubstrateNLobe = 0.5
    substrateNLobeConnection = calc_cubic_Bezier_pos(nSubunitV1, nSubunitCP1ForV1ToV2, nSubunitCP2ForV1ToV2, nSubunitV2, timePointSubstrateNLobe)
    timePointSubstrateCLobe = 0.35
    substrateCLobeConnection = calc_cubic_Bezier_pos(cSubunitV1, cSubunitCP1ForV1ToV2, cSubunitCP2ForV1ToV2, cSubunitV2, timePointSubstrateCLobe)

    # Generate the N subunit.
    nSubunitVerts = [nSubunitV1,  # Vertex 1.
                     nSubunitCP1ForV1ToV2,  # Control point 1 for movement between vertices 1 and 2.
                     nSubunitCP2ForV1ToV2,  # Control point 2 for movement between vertices 1 and 2.
                     nSubunitV2,  # Vertex 2.
                     nSubunitCP1ForV2ToV1,  # Control point 1 for movement between vertices 2 and 1.
                     nSubunitCP2ForV2ToV1,  # Control point 2 for movement between vertices 2 and 1.
                     nSubunitV1  # Vertex 1.
                    ]
    nSubunitMoves = [mpath.Path.MOVETO,
                     mpath.Path.CURVE4,
                     mpath.Path.CURVE4,
                     mpath.Path.CURVE4,
                     mpath.Path.CURVE4,
                     mpath.Path.CURVE4,
                     mpath.Path.CURVE4
                    ]
    nSubunitPath = mpath.Path(nSubunitVerts, nSubunitMoves)
    nSubunitPatch = patches.PathPatch(nSubunitPath)

    # Generate the C subunit.
    cSubunitVerts = [cSubunitV1,  # Vertex 1.
                     cSubunitCP1ForV1ToV2,  # Control point 1 for movement between vertices 1 and 2.
                     cSubunitCP2ForV1ToV2,  # Control point 2 for movement between vertices 1 and 2.
                     cSubunitV2, # Vertex 2.
                     cSubunitCP1ForV2ToV1,  # Control point 1 for movement between vertices 2 and 1.
                     cSubunitCP2ForV2ToV1,  # Control point 2 for movement between vertices 2 and 1.
                     cSubunitV1,  # Vertex 1.
                    ]
    cSubunitMoves = [mpath.Path.MOVETO,
                     mpath.Path.CURVE4,
                     mpath.Path.CURVE4,
                     mpath.Path.CURVE4,
                     mpath.Path.CURVE4,
                     mpath.Path.CURVE4,
                     mpath.Path.CURVE4
                    ]
    cSubunitPath = mpath.Path(cSubunitVerts, cSubunitMoves)
    cSubunitPatch = patches.PathPatch(cSubunitPath)

    # Generate the hinge region patch.
    hingeMidPoint = (hingeStartX - (0.05 * scale), hingeStartY - (0.15 * scale))
    hingeVerts = [(hingeStartX, hingeStartY),  # Hinge start.
                  (hingeStartX - (0.1 * scale), hingeStartY - (0.05 * scale)),
                  hingeMidPoint,
                  (hingeStopX - (0.1 * scale), hingeStopY + (0.05 * scale)),
                  (hingeStopX, hingeStopY)  # Hinge end.
                 ]
    hingeMoves = [mpath.Path.MOVETO,
                  mpath.Path.LINETO,
                  mpath.Path.LINETO,
                  mpath.Path.LINETO,
                  mpath.Path.LINETO
                 ]
    hingePath = mpath.Path(hingeVerts, hingeMoves)
    hingePatch = patches.PathPatch(hingePath, facecolor='none')

    # Generate the positions of the labels of the kinase's parts.
    labels = {}
    labels['N lobe'] = (nSubunitV1[0] - (0.4 * scale), nSubunitV1[1] - (0.1 * scale))
    labels['C lobe'] = (cSubunitV2[0] - (0.5 * scale), cSubunitV2[1])
    labels['Hinge'] = (hingeMidPoint[0] - (0.1 * scale), hingeMidPoint[1])

    return nSubunitPatch, cSubunitPatch, hingePatch, substrateNLobeConnection, substrateCLobeConnection, labels


def calc_cubic_Bezier_pos(p0, p1, p2, p3, t):
    """Calculate the X and Y coordinates of a position along a cubic Bezier curve.

    :param p0:      The X and Y coordinates of the first vertex defining the curve (p0[0] is the X coordinate).
    :type p0:       list of two floats
    :param p1:      The X and Y coordinates of the second vertex defining the curve (p1[0] is the X coordinate).
    :type p1:       list of two floats
    :param p2:      The X and Y coordinates of the third vertex defining the curve (p2[0] is the X coordinate).
    :type p2:       list of two floats
    :param p3:      The X and Y coordinates of the fouth vertex defining the curve (p3[0] is the X coordinate).
    :type p3:       list of two floats
    :param t:       The fraction along the curve at which to calculate the coordinates.
    :type t:        float (from 0 to 1)
    :returns :      The X and Y coordinates of the point the specified distance along the curve.
    :type :         float, float

    """

    x = (
         (((1 - t) ** 3) * p0[0]) +
         (3 * ((1 - t) ** 2) * t * p1[0]) +
         (3 * (1 - t) * (t ** 2) * p2[0]) +
         ((t ** 3) * p3[0])
        )
    y = (
         (((1 - t) ** 3) * p0[1]) +
         (3 * ((1 - t) ** 2) * t * p1[1]) +
         (3 * (1 - t) * (t ** 2) * p2[1]) +
         ((t ** 3) * p3[1])
        )
    return x, y


def substrate(nLobeConnection, cLobeConnection, scale=1.0):
    """Generate the objects needed to draw the kinase's substrate.

    :param nLobeConnection:     The X and Y coordinate of the position on the kinase's N subunit where the substrate connects to it (nLobeConnection[0] is the X coordinate).
    :type nLobeConnection:      list of two floats
    :param cLobeConnection:     The X and Y coordinate of the position on the kinase's C subunit where the substrate connects to it (cLobeConnection[0] is the X coordinate)
    :type cLobeConnection:      list of two floats
    :param scale:               How much to scale the base substrate.
    :type scale:                float
    :returns :                  The substrate, the point where a phosphate would connect to it and the coordinates of its labels.
    :type :                     matplotlib.patches object, list of two floats, tuple of two floats

    """

    # Generate the substrate.
    nLobeConnectionX = nLobeConnection[0]
    nLobeConnectionY = nLobeConnection[1]
    cLobeConnectionX = cLobeConnection[0]
    cLobeConnectionY = cLobeConnection[1]
    cp1ForNToC = (nLobeConnectionX - (0.1 * scale), cLobeConnectionY + (0.15 * scale))
    cp2ForNToC = (nLobeConnectionX - (0.15 * scale), cLobeConnectionY + (0.1 * scale))
    cp1ForCToN = (cLobeConnectionX + (0.6 * scale), cLobeConnectionY - (0.2 * scale))
    cp2ForCToN = (nLobeConnectionX + (0.2 * scale), nLobeConnectionY + (0.6 * scale))
    verts = [(nLobeConnectionX, nLobeConnectionY),
             cp1ForNToC,
             cp2ForNToC,
             (cLobeConnectionX, cLobeConnectionY),
             cp1ForCToN,
             cp2ForCToN,
             (nLobeConnectionX, nLobeConnectionY)
            ]
    moves = [mpath.Path.MOVETO,
             mpath.Path.CURVE4,
             mpath.Path.CURVE4,
             mpath.Path.CURVE4,
             mpath.Path.CURVE4,
             mpath.Path.CURVE4,
             mpath.Path.CURVE4
            ]
    path = mpath.Path(verts, moves)
    substrate = patches.PathPatch(path)

    # Determine the phosphate attachment point.
    phosphateConnectionPoint = calc_cubic_Bezier_pos(nLobeConnection, cp1ForNToC, cp2ForNToC, cLobeConnection, 0.6)

    # Determine the position of the label.
    labelPos = (nLobeConnection[0] - (0.1 * scale), nLobeConnection[1])

    return substrate, phosphateConnectionPoint, labelPos