import matplotlib.path as mpath

def main(startingXCoord, active=False, scale=1):
    """Generate the vertices and paths required to draw an ion channel.

    :param startingXCoord:  The X coordinate at which the drawing of the left subunit starts, and all other vertices are derived from.
    :type startingXCoord:   float
    :param active:          Whether the ion channel is being drawn active (True) or inactive (False).
    :type active:           boolean
    :param scale:           The scaling of the ion channel.
    :type scale:            float
    :returns :              The vertices and paths required to draw an ion channel.
    :type :                 list of tuples, list of tuples, list of matplotlib.path objects

    """

    mirrorX = startingXCoord + (0.23 * scale)  # The X coordinate to mirror the subunits around.

    # Left subunit vertices.
    if active:
        leftSubunitVertices = [(startingXCoord, 0.1 * scale),  # First vertex.
                               (startingXCoord, 0.95 * scale),  # Second vertex.
                               (startingXCoord, 1.1 * scale),  # Control point 1 for movement from second to third vertex.
                               (startingXCoord + (0.05 * scale), 1.125 * scale),  # Control point 2 for movement from second to third vertex.
                               (startingXCoord + (0.15 * scale), 0.95 * scale),  # Third vertex.
                               (startingXCoord + (0.3 * scale), 0.75 * scale),  # Control point 1 for movement from third to fourth vertex.
                               (startingXCoord + (0.05 * scale), 0.5 * scale),  # Control point 2 for movement from third to fourth vertex.
                               (startingXCoord + (0.15 * scale), 0.3 * scale),  # Fourth vertex.
                               (startingXCoord + (0.25 * scale), 0.05 * scale),  # Control point 1 for movement from fourth to first vertex.
                               (startingXCoord, -0.2 * scale),  # Control point 2 for movement from fourth to first vertex.
                               (startingXCoord, 0.1 * scale)  # First vertex.
                              ]
    else:
        leftSubunitVertices = [(startingXCoord, 0.1 * scale),  # First vertex.
                               (startingXCoord, 0.95 * scale),  # Second vertex.
                               (startingXCoord, 1.1 * scale),  # Control point 1 for movement from second to third vertex.
                               (startingXCoord + (0.05 * scale), 1.125 * scale),  # Control point 2 for movement from second to third vertex.
                               (startingXCoord + (0.15 * scale), 0.95 * scale),  # Third vertex.
                               (startingXCoord + (0.3 * scale), 0.75 * scale),  # Control point 1 for movement from third to fourth vertex.
                               (startingXCoord + (0.05 * scale), 0.5 * scale),  # Control point 2 for movement from third to fourth vertex.
                               (startingXCoord + (0.15 * scale), 0.3 * scale),  # Fourth vertex.
                               (startingXCoord + (0.35 * scale), 0.05 * scale),  # Control point 1 for movement from fourth to first vertex.
                               (startingXCoord + (0.0 * scale), -0.2 * scale),  # Control point 2 for movement from fourth to first vertex.
                               (startingXCoord + (0.0 * scale), 0.1 * scale)  # First vertex.
                              ]

    # Right subunit vertices.
    rightSubunitVertices = [((mirrorX - i[0]) + mirrorX, i[1]) for i in leftSubunitVertices]

    # Movements.
    movements = [mpath.Path.MOVETO,
                 mpath.Path.LINETO,
                 mpath.Path.CURVE4,
                 mpath.Path.CURVE4,
                 mpath.Path.CURVE4,
                 mpath.Path.CURVE4,
                 mpath.Path.CURVE4,
                 mpath.Path.CURVE4,
                 mpath.Path.CURVE4,
                 mpath.Path.CURVE4,
                 mpath.Path.CURVE4
                ]

    return leftSubunitVertices, rightSubunitVertices, movements