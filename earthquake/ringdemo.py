import matplotlib.animation as animation
import matplotlib.pyplot as plt
import pandas

import rings

# Setup the path to ffmpeg.
plt.rcParams['animation.ffmpeg_path'] = '/path/to/ffmpeg/executable'

# Setup the figure to be a blank white background.
fig = plt.figure(None)
ax = fig.gca()
ax.axis('equal')
ax.set_xlim([-50, 50])
ax.set_ylim([-50, 50])
ax.get_xaxis().set_visible(False)  # Hide the X axis.
ax.get_yaxis().set_visible(False)  # Hide the Y axis.
# Remove spines.
for i in ['left', 'right', 'top', 'bottom']:
    ax.spines[i].set_visible(False)

# Define the parameters for the individual rings.
r1 = {'Start' : 0, 'XCoord' : 0, 'YCoord' : 0, 'Radius' : 20, 'NumberOfRings' : 10, 'ColorsToUse' : 'red,black', 'Fade' : True, 'FadeStart' : 1, 'FadeEnd' : 0.1, 'UpdateSpeed' : 1}
r2 = {'Start' : 5, 'XCoord' : 20, 'YCoord' : 20, 'Radius' : 5, 'NumberOfRings' : 10, 'ColorsToUse' : 'blue,black', 'Fade' : True, 'FadeStart' : 1, 'FadeEnd' : 0.1, 'UpdateSpeed' : 1}
r3 = {'Start' : 0, 'XCoord' : -25, 'YCoord' : -20, 'Radius' : 10, 'NumberOfRings' : 10, 'ColorsToUse' : 'red,black', 'Fade' : True, 'FadeStart' : 1, 'FadeEnd' : 0.1, 'UpdateSpeed' : 2}
r4 = {'Start' : 10, 'XCoord' : -20, 'YCoord' : 25, 'Radius' : 15, 'NumberOfRings' : 5, 'ColorsToUse' : 'purple,blue', 'Fade' : True, 'FadeStart' : 0.25, 'FadeEnd' : 1, 'UpdateSpeed' : 1}
r5 = {'Start' : 0, 'XCoord' : 25, 'YCoord' : -25, 'Radius' : 10, 'NumberOfRings' : 8, 'ColorsToUse' : 'green,yellow', 'Fade' : True, 'FadeStart' : 0.5, 'FadeEnd' : 0.1, 'UpdateSpeed' : 3}
r6 = {'Start' : 15, 'XCoord' : 0, 'YCoord' : -30, 'Radius' : 12, 'NumberOfRings' : 14, 'ColorsToUse' : 'red,orange,yellow,green,blue,indigo,violet', 'Fade' : False, 'FadeStart' : 0.5, 'FadeEnd' : 0.1, 'UpdateSpeed' : 1}
parameters = pandas.DataFrame([r1, r2, r3, r4, r5, r6])
parameters['NumberOfTimeSteps'] = parameters['NumberOfRings'] * parameters['UpdateSpeed'] * 2  # Ensure that the waves go out to edge and immediately stop generating from middle.
rc = rings.RingCollection(parameters)

# In order for the animation to work, each of the objects that will be animated must be created and added to the figure first.
for i in rc.get_rings():
    ax.add_patch(i)

# Animate and save.
animinatedCollection = animation.FuncAnimation(fig, rc.update, frames=45, blit=True, interval=250, init_func=rc.get_rings, repeat=False)
FFwriter = animation.FFMpegWriter()
animinatedCollection.save('AnimatedRingCollection.mp4', writer = FFwriter, extra_args=['-vcodec', 'libx264'])