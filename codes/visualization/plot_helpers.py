import constants
import matplotlib.transforms as transforms

def label_panels_mosaic(fig, axes, xloc=0, yloc=1.0, size=constants.BIGGER_SIZE):
    """
    Labels the panels in a mosaic plot.

    Parameters:
        - fig: The figure object.
        - axes: A dictionary of axes objects representing the panels.
        - xloc: The x-coordinate for the label position (default: 0).
        - yloc: The y-coordinate for the label position (default: 1.0).
        - size: The font size of the labels (default: constants.BIGGER_SIZE).
    """
    for key in axes.keys():
        # label physical distance to the left and up:
        ax = axes[key]
        trans = transforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
        ax.text(xloc, yloc, key, transform=ax.transAxes + trans,
                fontsize=size, va='bottom')