import matplotlib.patches as patches
import matplotlib.path as mpath


def main(startX, startY, scale=1.0):
	# N subunit points.
	nSubunitV1 = (startX, startY)
	nSubunitCP1ForV1ToV2 = (startX + (0.8 * scale), startY + (0.2 * scale))
	nSubunitCP2ForV1ToV2 = (startX + (0.8 * scale), startY - (0.8 * scale))
	nSubunitV2 = (startX - (0.25 * scale), startY - (0.65 * scale))
	nSubunitCP1ForV2ToV1 = (startX - (0.5 * scale), startY - (0.6 * scale))
	nSubunitCP2ForV2ToV1 = (startX - (0.5 * scale), startY + (-0.1 * scale))
	
	# C subunit points.
	cSubunitV1 = (startX - (0.15 * scale), startY - (0.6 * scale))
	cSubunitCP1ForV1ToV2 = (startX + (1.0 * scale), startY - (0.35 * scale))
	cSubunitCP2ForV1ToV2 = (startX + (1.0 * scale), startY - (1.35 * scale))
	cSubunitV2 = (startX + (0.25 * scale), startY - (1.45 * scale))
	cSubunitCP1ForV2ToV1 = (startX - (0.45 * scale), startY - (1.55 * scale))
	cSubunitCP2ForV2ToV1 = (startX - (0.45 * scale), startY - (0.65 * scale))
	
	# Calculate the hinge points.
	timePointHingeStart = 0.25
	hingeStartX, hingeStartY = calc_cubic_Bezier_pos(nSubunitV2, nSubunitCP1ForV2ToV1, nSubunitCP2ForV2ToV1, nSubunitV1, timePointHingeStart)
	timePointHingeEnd = 0.65
	hingeStopX, hingeStopY = calc_cubic_Bezier_pos(cSubunitV2, cSubunitCP1ForV2ToV1, cSubunitCP2ForV2ToV1, cSubunitV1, timePointHingeEnd)
	
	# Calculate the substrate connection points.
	timePointSubstrateNLobe = 0.5
	substrateNLobeConnection = calc_cubic_Bezier_pos(nSubunitV1, nSubunitCP1ForV1ToV2, nSubunitCP2ForV1ToV2, nSubunitV2, timePointSubstrateNLobe)
	timePointSubstrateCLobe = 0.35
	substrateCLobeConnection = calc_cubic_Bezier_pos(cSubunitV1, cSubunitCP1ForV1ToV2, cSubunitCP2ForV1ToV2, cSubunitV2, timePointSubstrateCLobe)

	# N subunit.
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

	# C subunit.
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
	
	# Hinge region patch.
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

	labels = {}
	labels['N lobe'] = (nSubunitV1[0] - (0.4 * scale), nSubunitV1[1] - (0.1 * scale))
	labels['C lobe'] = (cSubunitV2[0] - (0.5 * scale), cSubunitV2[1])
	labels['Hinge'] = (hingeMidPoint[0] - (0.1 * scale), hingeMidPoint[1])

	return nSubunitPatch, cSubunitPatch, hingePatch, substrateNLobeConnection, substrateCLobeConnection, labels


def calc_cubic_Bezier_pos(p0, p1, p2, p3, t):
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
	
	labelPos = (nLobeConnection[0] - (0.1 * scale), nLobeConnection[1])

	return substrate, phosphateConnectionPoint, labelPos