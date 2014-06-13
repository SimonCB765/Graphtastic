import matplotlib.patches as patches
import numpy as np


def main(startX, startY, baseRiboseRadius=0.2, basePhosphateRadius=0.1, scale=1.0, numberOfPhosphates=3):
	"""
	"""
	
	# Default orientation is for vertex a to be directly above the center. Ordering the vertices clockwise, the angle between vertex b, the center
	# and the horizontal is 90 - angleOfRotation, as there is a 90 degree angle formed between vertex a, the center and the horizontal.
	riboseRadius = scale * baseRiboseRadius
	angleOfRotation = (72 * np.pi) / 180  # 72 degrees
	vertexBCenterHorizontalAngle = (np.pi / 2) - angleOfRotation
	ribose = patches.RegularPolygon((startX, startY), numVertices=5, radius=riboseRadius, orientation=-vertexBCenterHorizontalAngle)

	phosphates = []
	phosphateRadius = scale * basePhosphateRadius
	for i in range(numberOfPhosphates):
		phosphates.append(patches.Circle((startX + riboseRadius + ((1 + (2 * i)) * phosphateRadius), startY), phosphateRadius))

	return ribose, phosphates