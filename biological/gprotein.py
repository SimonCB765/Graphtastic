import matplotlib.patches as patches
import numpy as np

def main(gAlphaCenter, scale=1.0, label=False, GTP=False, disassociate=False):
    """Generate the patches required to draw a G protein, either associated or disassociated.

    :param gAlphaCenter:    The center of the g alpha subunit. All other subunit positions will be based on this center.
    :type gAlphaCenter:     Tuple or list of two elements. The first element should be the X coordinate and the second the Y coordinate.
    :param scale:           The scaling factor for the G protein. Controls the size of the final protein.
    :type scale:            float
    :param label:           Whether label locations for the subunits and GTP should be returned.
    :type label:            boolean
    :param GTP:             Whether the GTP should be drawn as well.
    :type GTP:              boolean
    :param disassociate:    Whether the G protein is associated or disassociated.
    :type disassociate:     boolean
    :returns :              The three subunits of the G proteins, along with a GTP if required.
    :type :                 matplotlib.patch, matplotlib.patch, matplotlib.patch, matplotlib.patch or None

    """

    # Determine the radii of the elements of the protein.
    gAlphaRadius = 0.35 * scale  # The scaled radius of the alpha subunit.
    gBetaRadius = 0.2 * scale  # The scaled radius of the beta subunit.
    gGammaRadius = 0.2 * scale  # The scaled radius of the gamma subunit
    baseGTPRadius = 0.08  # The base radius of the GTP molecule.
    GTPRadius = baseGTPRadius * scale  # The scaled radius of the GTP molecule.

    # Determine the spatial relationship of the individual subunits.
    gGammaConnectionWithAlpha = (gAlphaCenter[0] + (gAlphaRadius * np.cos(30 * np.pi / 180)), gAlphaCenter[1] + (gAlphaRadius * np.sin(30 * np.pi / 180)))
    gGammaCenter = (gGammaConnectionWithAlpha[0] + gGammaRadius + (gGammaRadius if disassociate else 0), gGammaConnectionWithAlpha[1])
    gBetaCenter = (gGammaCenter[0], gGammaCenter[1] - gBetaRadius - gGammaRadius)

    # Generate the subunit patches.
    gAlpha = patches.Circle(gAlphaCenter, gAlphaRadius)
    gBeta = patches.Circle(gBetaCenter, gBetaRadius)
    gGamma = patches.Circle(gGammaCenter, gGammaRadius)

    # Generate the GTP patch if required.
    patchGTP = None
    if GTP:
        GTPConnectionWithGAlpha = (gAlphaCenter[0] - (gAlphaRadius * np.sin(30 * np.pi / 180)), gAlphaCenter[1] - (gAlphaRadius * np.cos(30 * np.pi / 180)))
        patchGTP = patches.Circle(GTPConnectionWithGAlpha, GTPRadius)

    # Return any desired label coordinates.
    labelCoordinates = {}
    if label:
        labelCoordinates['alpha'] = [gAlphaCenter[0], gAlphaCenter[1]]
        labelCoordinates['beta'] = [gBetaCenter[0], gBetaCenter[1]]
        labelCoordinates['gamma'] = [gGammaCenter[0], gGammaCenter[1]]
        labelCoordinates['GTP'] = [GTPConnectionWithGAlpha[0], GTPConnectionWithGAlpha[1] - (baseGTPRadius * 1.5)]

    return gAlpha, gBeta, gGamma, patchGTP, labelCoordinates