import datetime
import math
import matplotlib.animation as animation
from mpl_toolkits.basemap import Basemap
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import pandas

import rings

# Setup the path to ffmpeg.
plt.rcParams['animation.ffmpeg_path'] = '/path/to/ffmpeg/executable'

# Parse the earthquake data.
fileRawData = 'centennial_Y2K.CAT'
earthquakeParsedData = []
with open(fileRawData, 'r') as readFile:
    # The fields that we are interested in are the year, month, day, latitude, longitude, magnitude and depth of the earthquake.
    for line in readFile:
        quake = {}
        quake['Year'] = int(line[12:17])  # The year is always four numbers in width.
        quake['Month'] = int(line[17:19].strip())  # The month can be one or two numbers (e.g. January is 1 not 01).
        quake['Day'] = int(line[20:22].strip())  # The day can be one or two numbers (e.g. the first of the month is 1 not 01).
        quake['Latitude'] = float(line[37:44].strip())  # Latitude ranges from -90 to 90 with 3 decimal places.
        quake['Longitude'] = float(line[44:52].strip())  # Longitdue ranges from -180 to 180 with 3 decimal places.
        quake['Depth'] = float(line[53:58].strip())  # The depth is a float no greater than 999.0.
        quake['Magnitude'] = float(line[67:70].strip())  # The magnitude is between 5.5 and 9.9.
        earthquakeParsedData.append(quake)
earthquakes = pandas.DataFrame(earthquakeParsedData)
earthquakes = earthquakes.sort(['Year', 'Month', 'Day'], ascending=True)  # Sort quakes by year then month then day.

# Shift any days that are marked as day 0 to be day 1.
earthquakes['Day'][earthquakes['Day'] == 0] = 1

# Add a scaled magnitude column.
minMagnitude = 5.0
maxMagnitude = earthquakes.max()['Magnitude']
scaledMagnitudes = (earthquakes['Magnitude'] - minMagnitude) / (maxMagnitude - minMagnitude)
earthquakes['ScaledMagnitude'] = scaledMagnitudes

# Add a column indicating how many years after the first earthquake each earthquake occurred.
yearOfFirstQuake = 1900
earthquakes['OrderedYears'] = earthquakes['Year'] - yearOfFirstQuake

# Add a column indicating how many months after the first earthquake each earthquake occurred.
monthOfFirstQuake = 1900 + (1 / 12)  # The first earthquake recorded is in January 1900.
earthquakes['OrderedMonths'] = (((earthquakes['Year'] + (earthquakes['Month'] / 12) - monthOfFirstQuake) * 12) - 1).round()  # -1 as January is recorded as 1 not 0.

# Setup the map.
figure = plt.figure(figsize=(18,12))  # Set the size of the map to a size where the detail is clearer.
axis = figure.add_axes([0, 0, 1, 1])  # Create an axis to take up the whole figure.
map = Basemap(projection='cyl', llcrnrlat=-90, urcrnrlat=90, llcrnrlon=-180, urcrnrlon=180, resolution='l', area_thresh = 1000.0)
map.drawcoastlines(color='white', linewidth=0.5)
map.drawcountries(color='0.5')
map.fillcontinents(color='black', lake_color='black')
map.drawmapboundary(fill_color='black')  # This will set the background color (i.e. the ocean in this case) to black.

# Determine the date of the first earthquake.
firstQuake = earthquakes.head(1)
firstQuakeDate = datetime.date(firstQuake['Year'], firstQuake['Month'], firstQuake['Day'])

# Define the parameters for the rings.
params = []
framesPerYear = 30  # Animate by years, as animating by months or days makes for a long and large animation.
#framesPerMonth = 20
maxRadius = 11
updateSpeed = 1
maxRings = 15
#maxRings = framesPerMonth / (2 * updateSpeed)
#colorPicker = lambda x : 'limegreen' if x < 0.5 else ('yellow' if x < 0.75 else 'red')
colorPicker = lambda x : 'limegreen,none' if x < 0.5 else ('yellow,none' if x < 0.75 else 'red,none')
for index, row in earthquakes.iterrows():
	if row['Year'] > 1909:
		break
	ringStart = int(row['OrderedYears']) * framesPerYear
	#ringStart = int(row['OrderedMonths']) * framesPerMonth
	ringXCoord = row['Longitude']
	ringYCoord = row['Latitude']
	scaledMagnitude = row['ScaledMagnitude']
	ringRadius = math.ceil(maxRadius * scaledMagnitude)
	ringNumber = maxRings#int(math.ceil(maxRings * scaledMagnitude))
	ringTimeSteps = ringNumber * updateSpeed * 2
	ringColors = colorPicker(row['ScaledMagnitude'])  # Color code earthquakes by magnitude.
	ringFade = False
	ringFadeStart = 1#scaledMagnitude
	ringFadeEnd = 0.5
	ringParams = {'Start' : ringStart, 'XCoord' : ringXCoord, 'YCoord' : ringYCoord, 'Radius' : ringRadius, 'NumberOfRings' : ringNumber, 'NumberOfTimeSteps' : ringTimeSteps, 'ColorsToUse' : ringColors, 'Fade' : ringFade, 'FadeStart' : ringFadeStart, 'FadeEnd' : ringFadeEnd, 'UpdateSpeed' : updateSpeed}
	params.append(ringParams)
parameters = pandas.DataFrame(params)
rc = rings.RingCollection(parameters, 'Disperse')
#rc = rings.RingCollection(parameters, 'FadeOut')

# In order for the animation to work, each of the objects that will be animated must be created and added to the figure first.
# In order for the colors of the larger magnitude earthquakes to be easily visible, place them on the axis last. This will ensure
# that they are on top of the smaller magnitude earthquakes in terms of their visibility.
sortedRings = sorted(zip(earthquakes['ScaledMagnitude'], rc.get_ring_sets()))  # Sorted rings by decreasing magnitude.
for i, j in sortedRings:
	for k in j.get_rings():
		axis.add_patch(k)

# Animate the earthquakes.
frames = int(earthquakes.max()['OrderedYears'] * framesPerYear)
#frames = int(earthquakes.max()['OrderedMonths'] * framesPerMonth)
frames = int(parameters.max()['Stop'])
animinatedCollection = animation.FuncAnimation(figure, rc.update, frames=frames, blit=True, init_func=rc.get_rings, repeat=False)

# Animate a date time stamp.
font = FontProperties()
font.set_style('italic')
font.set_family('gabriola')  # Can use matplotlib.font_manager.findSystemFonts(fontpaths=None) to find a suitable font name.
timeStamp = axis.text(axis.get_xlim()[0], axis.get_ylim()[0], '', color='white', fontproperties=font, fontsize=60, horizontalalignment='left', verticalalignment='bottom')
def text_init():
	return timeStamp
def text_update_by_year(frameNumber):
	yearsFromStart = frameNumber // framesPerYear
	currentYear = yearOfFirstQuake + yearsFromStart
	timeStamp.set_text('{0}'.format(currentYear))
	return timeStamp
animinatedText = animation.FuncAnimation(figure, text_update_by_year, frames=frames, blit=True, init_func=text_init, repeat=False)
#months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
#monthOfFirstQuake = 0
#yearOfFirstQuake = 1900
#def text_update_by_month(frameNumber):
#	monthsFromStart = frameNumber // framesPerMonth
#	yearsFromStart = monthsFromStart // 12
#	currentMonth = months[monthOfFirstQuake + (monthsFromStart % 12)]
#	currentYear = yearOfFirstQuake + yearsFromStart
#	timeStamp.set_text('{0}, {1}'.format(currentYear, currentMonth))
#	return timeStamp
#animinatedText = animation.FuncAnimation(figure, text_update_by_month, frames=frames, blit=True, init_func=text_init, repeat=False)

# Save the animations.
FFwriter = animation.FFMpegWriter()
animinatedCollection.save('AnimatedEarthquakes-Disperse.mp4', writer = FFwriter, fps=60, extra_args=['-vcodec', 'libx264'], extra_anim=[animinatedText], savefig_kwargs={'facecolor' : 'black'})