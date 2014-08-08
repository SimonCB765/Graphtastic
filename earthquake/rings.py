from collections import deque
import matplotlib.patches as patches
from matplotlib.path import Path
import matplotlib.pyplot as plt
import numpy as np

class DispersionRings:
    """Class to generate animated concentric rings that disperse."""

    def __init__(self, xCoord, yCoord, totalRadius=10, numberOfRings=10, numberOfTimeSteps=25, colorsToUse=['black'], fade=False, fadeStart=1.0, fadeEnd=0.1,
                 updateSpeed=1):
        """Initialises the rings.

        :param xCoord:              The X coordinate at which the set of rings will be centered.
        :type xCoord:               float
        :param yCoord:              The Y coordinate at which the set of rings will be centered.
        :type yCoord:               float
        :param totalRadius:         The radius of the outermost ring form the center of the set of rings.
        :type totalRadius:          float
        :param numberOfRings:       The number of rings to create.
        :type numberOfRings:        int
        :param numberOfTimeSteps:   The number of time steps after which the rings will contain no color (and therefore will disappear).
        :type numberOfTimeSteps:    int
        :param colorsToUse:         The colors to use in coloring the rings. The set of colors chosen will be cycled through, so that if the colors
                                    are ['red', 'green', 'blue'], ring 0 will be blue, ring 1 red, ring 2 green, ring 3 blue, etc. If you want a gap
                                    between colors then use 'none' as a color (e.g. alternating black and clear rings would be ['black', 'none']).
        :type colorsToUse:          list of valid colors for matplotlib.patches objects
        :param fade:                Whether the rings should have different transparencies as you move away from the center.
        :type fade:                 boolean
        :param fadeStart:           The alpha value for the innermost ring.
        :type fadeStart:            float
        :param fadeEnd:             The alpha value for the outermost ring.
        :type fadeEnd:              float
        :param updateSpeed:         Controls the speed of ring updating. E.g. if updateSpeed == 1 the rings will update every frame, if updateSpeed == 2
                                    the rings will update every other frame.
        :type updateSpeed:          int

        """

        self.xCoord = xCoord
        self.yCoord = yCoord
        self.totalRadius = totalRadius
        self.numberOfRings = numberOfRings
        self.ringWidth = totalRadius / numberOfRings
        self.numberOfTimeSteps = numberOfTimeSteps
        self.colorsToUse = colorsToUse
        self.fade = fade
        self.fadeStart = fadeStart
        self.fadeEnd = fadeEnd
        self.updateSpeed = updateSpeed

        # Set up variables for controlling the color of the next inner ring generated.
        self.currentColorIndex = 0  # The index of the color to make the next inner ring
        self.currentColor = self.colorsToUse[self.currentColorIndex]  # The color to make the next inner ring.

        # Determine when to stop generating new colored rings from the centre. As the rings propagate out from the center of the set of rings, this time
        # step must be calculated such that the last colored ring disappears past the final ring after numberOfTimeSteps time steps. For example, if
        # numberOfTimeSteps == 40, then the the 39th update must make it so that only the outermost ring of the collection is colored, and the 40th update
        # will cause all rings to be clear (and therefore to disappear).
        self.stopGenerating = self.numberOfTimeSteps - (self.numberOfRings * self.updateSpeed)  # Once this number is reached no new colors will be generated from the center.

        # Setup the alpha value for each ring, and the queue used to keep track of each ring's current color
        self.colors = deque(['none'] * self.numberOfRings)
        if fade:
            fadeStep = (fadeEnd - fadeStart) / self.numberOfRings
            self.fadeValues = [(fadeStart + (fadeStep * i)) for i in range(self.numberOfRings)]
        else:
            self.fadeValues = [1] * self.numberOfRings

        # Create the rings by first generating a set of concentric circles ordered so that the one with the largest radius is at index 0.
        circles = [patches.Circle((self.xCoord, self.yCoord), radius=(self.ringWidth * i)) for i in range(1, self.numberOfRings + 1)[::-1]]

        # Create the individual rings. For each ring this is done by taking the path of the circle at index i - 1, reversing it, and then
        # appending it to the path of the (larger) circle at index i. This ensures that the patch generated from this path will only be the
        # portion of the circle at index i that is not overlapped by the circle at index i - 1.
        circlePaths = [i.get_transform().transform_path(i.get_path()) for i in circles]
        circleVertsCW = [i.vertices[:-1] for i in circlePaths]  # Remove close poly vertex at the end of each list of vertices.
        circleVertsCCW = [i.vertices[:-1][::-1] for i in circlePaths]  # Remove close poly vertex and reverse vertices.
        circleCodes = [i.codes[:-1] for i in circlePaths]  # Remove close poly code at the end of each list of codes.
        ringVerts = [np.concatenate((circleVertsCW[i], circleVertsCCW[i + 1])) for i in range(self.numberOfRings - 1)] + [circleVertsCW[-1]]
        ringCodes = [np.concatenate((circleCodes[i], circleCodes[i + 1])) for i in range(self.numberOfRings - 1)] + [circleCodes[-1]]

        # Make the ring paths into patches.
        ringPaths = [Path(ringVerts[i], ringCodes[i]) for i in range(self.numberOfRings)]
        ringPaths = ringPaths[::-1]  # Reverse the ring paths so that the smaller inner rings are at the start of the list.
        self.ringPatches = [patches.PathPatch(j, alpha=self.fadeValues[i], facecolor=self.colors[i], edgecolor=self.colors[i], linewidth=0) for i,j in enumerate(ringPaths)]


    def get_rings(self):
        """Return the rings."""
        return self.ringPatches


    def update(self, frameNumber):
        """Called when the rings need updating.

        :param frameNumber:     The number of the current frame.
        :type frameNumber:      int

        """

        if (frameNumber >= self.numberOfTimeSteps) or (frameNumber % self.updateSpeed):
            # If no change should be made to the rings because they should stop animating or this is a frame when they should not update.
            return []
        else:
            if frameNumber >= self.stopGenerating:
                # Turn off the color.
                self.currentColor = 'none'
            else:
                # Choose the next color.
                self.currentColor = self.colorsToUse[self.currentColorIndex]
                self.currentColorIndex = (self.currentColorIndex + 1) % len(self.colorsToUse)

            # Update the colors.
            oldOuterColor = self.colors.pop()
            self.colors.appendleft(self.currentColor)
            for i,j in enumerate(self.ringPatches):
                j.set_color(self.colors[i])

        return self.ringPatches


class FadeOutRings:
    """Class to generate animated concentric rings that fade out."""

    def __init__(self, xCoord, yCoord, totalRadius=10, numberOfRings=10, numberOfTimeSteps=25, colorsToUse=['black'], fade=False, fadeStart=1.0, fadeEnd=0.1,
                 updateSpeed=1):
        """Initialises the rings.

        :param xCoord:              The X coordinate at which the set of rings will be centered.
        :type xCoord:               float
        :param yCoord:              The Y coordinate at which the set of rings will be centered.
        :type yCoord:               float
        :param totalRadius:         The radius of the outermost ring form the center of the set of rings.
        :type totalRadius:          float
        :param numberOfRings:       The number of rings to create.
        :type numberOfRings:        int
        :param numberOfTimeSteps:   The number of time steps after which the rings will contain no color (and therefore will disappear).
        :type numberOfTimeSteps:    int
        :param colorsToUse:         The colors to use in coloring the rings. The set of colors chosen will be cycled through, so that if the colors
                                    are ['red', 'green', 'blue'], ring 0 will be blue, ring 1 red, ring 2 green, ring 3 blue, etc. If you want a gap
                                    between colors then use 'none' as a color (e.g. alternating black and clear rings would be ['black', 'none']).
        :type colorsToUse:          list of valid colors for matplotlib.patches objects
        :param fade:                Whether the rings should have different transparencies as you move away from the center.
        :type fade:                 boolean
        :param fadeStart:           The alpha value for the innermost ring.
        :type fadeStart:            float
        :param fadeEnd:             The alpha value for the outermost ring.
        :type fadeEnd:              float
        :param updateSpeed:         Controls the speed of ring updating. E.g. if updateSpeed == 1 the rings will update every frame, if updateSpeed == 2
                                    the rings will update every other frame.
        :type updateSpeed:          int

        """

        self.xCoord = xCoord
        self.yCoord = yCoord
        self.totalRadius = totalRadius
        self.numberOfRings = numberOfRings
        self.ringWidth = totalRadius / numberOfRings
        self.numberOfTimeSteps = numberOfTimeSteps
        self.colorsToUse = colorsToUse
        self.fade = fade
        self.fadeStart = fadeStart
        self.fadeEnd = fadeEnd
        self.updateSpeed = updateSpeed

        # Set up variables for controlling the color of the next inner ring generated.
        self.currentColorIndex = 0  # The index of the color to make the next inner ring
        self.currentColor = self.colorsToUse[self.currentColorIndex]  # The color to make the next inner ring.

        # Determine when to stop generating new colored rings from the centre. As the rings propagate out from the center of the set of rings, this time
        # step must be calculated such that the rings disappear after numberOfTimeSteps time steps.
        self.stopGenerating = self.numberOfTimeSteps - (self.numberOfRings * self.updateSpeed)  # Once this number is reached no new colors will be generated from the center.

        # Setup the alpha value for each ring, and the queue used to keep track of each ring's current color
        self.colors = deque(['none'] * self.numberOfRings)
        if fade:
            fadeStep = (fadeEnd - fadeStart) / self.numberOfRings
            self.fadeValues = [(fadeStart + (fadeStep * i)) for i in range(self.numberOfRings)]
        else:
            self.fadeValues = [1] * self.numberOfRings
        fadeOutTimeSteps = self.numberOfTimeSteps - self.stopGenerating
        self.fadeDecreaseValues = [i / fadeOutTimeSteps for i in self.fadeValues]

        # Create the rings by first generating a set of concentric circles ordered so that the one with the largest radius is at index 0.
        circles = [patches.Circle((self.xCoord, self.yCoord), radius=(self.ringWidth * i)) for i in range(1, self.numberOfRings + 1)[::-1]]

        # Create the individual rings. For each ring this is done by taking the path of the circle at index i - 1, reversing it, and then
        # appending it to the path of the (larger) circle at index i. This ensures that the patch generated from this path will only be the
        # portion of the circle at index i that is not overlapped by the circle at index i - 1.
        circlePaths = [i.get_transform().transform_path(i.get_path()) for i in circles]
        circleVertsCW = [i.vertices[:-1] for i in circlePaths]  # Remove close poly vertex at the end of each list of vertices.
        circleVertsCCW = [i.vertices[:-1][::-1] for i in circlePaths]  # Remove close poly vertex and reverse vertices.
        circleCodes = [i.codes[:-1] for i in circlePaths]  # Remove close poly code at the end of each list of codes.
        ringVerts = [np.concatenate((circleVertsCW[i], circleVertsCCW[i + 1])) for i in range(self.numberOfRings - 1)] + [circleVertsCW[-1]]
        ringCodes = [np.concatenate((circleCodes[i], circleCodes[i + 1])) for i in range(self.numberOfRings - 1)] + [circleCodes[-1]]

        # Make the ring paths into patches.
        ringPaths = [Path(ringVerts[i], ringCodes[i]) for i in range(self.numberOfRings)]
        ringPaths = ringPaths[::-1]  # Reverse the ring paths so that the smaller inner rings are at the start of the list.
        self.ringPatches = [patches.PathPatch(j, alpha=self.fadeValues[i], facecolor=self.colors[i], edgecolor=self.colors[i], linewidth=0) for i,j in enumerate(ringPaths)]


    def get_rings(self):
        """Return the rings."""
        return self.ringPatches


    def update(self, frameNumber):
        """Called when the rings need updating.

        :param frameNumber:     The number of the current frame.
        :type frameNumber:      int

        """

        if (frameNumber >= self.numberOfTimeSteps) or (frameNumber % self.updateSpeed):
            # If no change should be made to the rings because they should stop animating or this is a frame when they should not update.
            return []
        elif frameNumber >= self.stopGenerating:
            # Fade the color out.
            self.fadeValues = [max(0, i - j) for i, j in zip(self.fadeValues, self.fadeDecreaseValues)]
            for i,j in enumerate(self.ringPatches):
                j.set_alpha(self.fadeValues[i])

        # Choose the next color.
        self.currentColor = self.colorsToUse[self.currentColorIndex]
        self.currentColorIndex = (self.currentColorIndex + 1) % len(self.colorsToUse)

        # Update the colors.
        oldOuterColor = self.colors.pop()
        self.colors.appendleft(self.currentColor)
        for i,j in enumerate(self.ringPatches):
            j.set_color(self.colors[i])

        return self.ringPatches


class RingCollection:
    """Class to generate a collection of animated rings."""

    def __init__(self, parameters, ringType='Disperse'):
        """Initialise the collection.

        Each row of the parameters DataFrame contains the parameters needed to create one DispersionRings object, along with the information to
        determine when the object should be created and deleted. The DataFrame must contain the following columns (see DispersionRings or more
        detail on the individual columns):
            Start               The time step at which the DispersionRings object should be initialised.
                                int
            XCoord              The X coordinate at which the DispersionRings object should be displayed.
                                float
            YCoord              The Y coordinate at which the DispersionRings object should be displayed.
                                float
            Radius              The radius from the center of the outermost ring.
                                float
            NumberOfRings       The number of rings to create.
                                int
            NumberOfTimeSteps   The number of time steps over which the DispersionRings object should be active.
                                int
            ColorsToUse         The colors for the rings.
                                comma delimited string
            Fade                Whether the DispersionRings object should change in transparency.
                                boolean
            FadeStart           The alpha value for the innermost ring.
                                float
            FadeEnd             The alpha value for the outermost ring.
                                float
            UpdateSpeed         The speed with which the rings will update.
                                int
        One final column, 'Stop', will be added to the DataFrame. Any other columns will be ignored.

        :param parameters:  The parameters used to create the ring sets in the collection.
        :type parameters:   pandas.DataFrame
        :param ringType:    The type of ring sets to create.
        :type ringType:     one of 'Disperse' or 'FadeOut'

        """

        self.parameters = parameters
        self.parameters['Stop'] = self.parameters['Start'] + self.parameters['NumberOfTimeSteps']
        self.ringsAlive = set([])  # Record of the rings currently up for animation.

        # Ideally rings would only be created when their turn to be animated came up. However, the matplotlib animation API requires that an animated object
        # be created and attached to the figure prior to the start of the animation. All rings must therefore be created at initialisation.
        self.allRings = {}
        if ringType == 'Disperse':
            for index, row in self.parameters.iterrows():
                createdRingSet = DispersionRings(row['XCoord'], row['YCoord'], row['Radius'], row['NumberOfRings'], row['NumberOfTimeSteps'],
                                               row['ColorsToUse'].split(','), row['Fade'], row['FadeStart'], row['FadeEnd'], row['UpdateSpeed'])
                self.allRings[index] = createdRingSet
        else:
            for index, row in self.parameters.iterrows():
                createdRingSet = FadeOutRings(row['XCoord'], row['YCoord'], row['Radius'], row['NumberOfRings'], row['NumberOfTimeSteps'],
                                              row['ColorsToUse'].split(','), row['Fade'], row['FadeStart'], row['FadeEnd'], row['UpdateSpeed'])
                self.allRings[index] = createdRingSet



    def get_ring_sets(self):
        """Return the sets of rings in the collection."""
        return [self.allRings[i] for i in self.allRings]


    def get_rings(self):
        """Return the rings in the collection."""
        return [j for i in self.allRings for j in self.allRings[i].get_rings()]


    def update(self, frameNumber):
        """Called when the rings need updating.

        :param frameNumber:     The number of the current frame.
        :type frameNumber:      int

        """

        # Delete rings that should end.
        oldRings = self.parameters[self.parameters['Stop'] == frameNumber]
        for index, row in oldRings.iterrows():
            del self.allRings[index]
            self.ringsAlive -= set([index])

        # Find all new rings to create.
        newRings = self.parameters[self.parameters['Start'] == frameNumber]
        for index, row in newRings.iterrows():
            self.ringsAlive |= set([index])

        updatedRings = []
        for i in self.ringsAlive:
            updatedRings.extend(self.allRings[i].update(frameNumber - self.parameters.iloc[i]['Start']))

        return updatedRings