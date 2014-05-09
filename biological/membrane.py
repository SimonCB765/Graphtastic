import matplotlib.path as mpath
import matplotlib.patches as patches

def main(topLayerY=1.0, bottomLayerY=0.0, startPointLeft=0.0, stopPointRight=10.0, headRadius=0.1):
	"""Generate the patches and movements required to draw a phospholipid bilayer.
	
	:param topLayerY:		The y location for the center of the phosphate heads in the top layer of the bilayer.
	:type topLayerY:		float
	:param bottomLayerY:	The y location for the center of the phosphate heads in the bottom layer of the bilayer.
	:type bottomLayerY:		float
	:param startPointLeft:	The x location for the leftmost edge of the leftmost phosphate head.
	:type startPointLeft:	float
	:param stopPointRight:	The x location beyond which no new phosphate heads should be created (the final head can end beyond this point).
	:type stopPointRight:	float
	:param headRadius:		The radius of each phosphate head.
	:type headRadius:		float
	:returns:				The phosphate heads, the vertices for the start and end of the lipid tails, the movements required to draw the tails.
	:type:					list of matplotlib.patches, list of tuples, list of mpath.Path objects
	"""

	heads = []  # The patches for the phosphate heads.
	tailVerts = []  # The vertices for the start and end of each lipid tail.
	tailMoves = []  # The movements between vertices needed to draw the lipid tails.

	intramembraneSpace = topLayerY - bottomLayerY - (2 * headRadius)  # The space between the phosphate heads.
	tailDistanceFromHead = (3 / 7) * intramembraneSpace  # The distance that each tail should travel vertically.

	currentLeftEdge = startPointLeft  # The left limit of the phosphate being drawn currently.
	while currentLeftEdge <= stopPointRight:
		# Keep generating phosphate heads and the vertices for the lipid tails until the stopping point is reached.
		currentCenter = currentLeftEdge + headRadius  # The current center of the phosphate head.
		
		# Generate the phosphate head for the top and bottom layers.
		heads.append(patches.Circle((currentCenter, topLayerY), radius=headRadius))
		heads.append(patches.Circle((currentCenter, bottomLayerY), radius=headRadius))
		
		# Generate the lipid tail vertices for the top and bottom layers.
		leftTailTopVerts = [(currentCenter, topLayerY), (currentCenter - (0.5 * headRadius), topLayerY - tailDistanceFromHead - headRadius)]
		tailVerts.append(leftTailTopVerts)
		leftTailBotVerts = [(currentCenter, bottomLayerY), (currentCenter - (0.5 * headRadius), bottomLayerY + tailDistanceFromHead + headRadius)]
		tailVerts.append(leftTailBotVerts)
		rightTailTopVerts = [(currentCenter, topLayerY), (currentCenter + (0.5 * headRadius), topLayerY - tailDistanceFromHead - headRadius)]
		tailVerts.append(rightTailTopVerts)
		rightTailBotVerts = [(currentCenter, bottomLayerY), (currentCenter + (0.5 * headRadius), bottomLayerY + tailDistanceFromHead + headRadius)]
		tailVerts.append(rightTailBotVerts)
		
		currentLeftEdge += headRadius * 2  # Increment the left edge for the next phosphate head.

	tailMoves = [[mpath.Path.MOVETO, mpath.Path.LINETO] for i in tailVerts]  # Create the movements needed to draw the lipid tails.
	return heads, tailVerts, tailMoves