import re
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib import font_manager
from matplotlib.legend import Legend

from seaborn import cm
from seaborn.axisgrid import Grid
from seaborn.utils import (despine, axis_ticklabels_overlap, relative_luminance, to_utf8)
from seaborn.axisgrid import Grid
from seaborn.utils import (despine, axis_ticklabels_overlap, relative_luminance, to_utf8)

## TODO: make legend formatting better
def fix_plot(axs, plot_dict=None, fontdir=None):
    
    '''
    Convenience function to change matplotlib formatting.
    default_plot_dict = {
                            'weight': 'regular',
                            'family': 'Helvetica',
                            'fontsize': 8,
                            'pad': 2,
                            'grid': False,
                            'x_rotation': 0,
                            'y_rotation': 0,
                            'legend': True,
                            'fixlegend': False,
                            'snsfig': True,
                            'rm_legend_title': False,
                            'legend_loc': 'upper left',
                            'markerscale': 0.2,
                            'frame': True,
                            'subplot_behavior': True
                        }
    '''

    if fontdir is not None:
        try:
            font_files = font_manager.findSystemFonts(fontpaths=fontdir)

            for font_file in font_files:
                font_manager.fontManager.addfont(font_file)
        except:
            pass

    ## default settings
    default_plot_dict = {
                            'weight': 'regular',
                            'family': 'Helvetica',
                            'fontsize': 8,
                            'pad': 2,
                            'grid': False,
                            'x_rotation': 0,
                            'y_rotation': 0,
                            'legend': True,
                            'fixlegend': False,
                            'snsfig': True,
                            'rm_legend_title': False,
                            'legend_loc': 'upper left',
                            'markerscale': 0.2,
                            'frame': True,
                            'subplot_behavior': True
                        }

    ## Update settings based on input dict
    for key in plot_dict.keys():
        default_plot_dict[key] = plot_dict[key]

    # set font
    plt.rcParams['font.family'] = default_plot_dict['family']
    plt.rcParams['font.weight'] = default_plot_dict['weight']

    _= axs.tick_params(axis= 'both', labelsize= default_plot_dict['fontsize'], pad= default_plot_dict['pad'])
    _= axs.grid(visible= default_plot_dict['grid'])
    _= axs.set_frame_on(default_plot_dict['frame'])

    if 'xlim' in default_plot_dict.keys():
        _= axs.set_xlim(default_plot_dict['xlim'])
    if 'ylim' in default_plot_dict.keys():
        _= axs.set_ylim(default_plot_dict['ylim'])
    if 'xticks' in default_plot_dict.keys():
        _= axs.set_xticks(default_plot_dict['xticks'])
    if 'xticklabels' in default_plot_dict.keys():
        _= axs.set_xticklabels(default_plot_dict['xticklabels'], fontsize= default_plot_dict['fontsize'], rotation= default_plot_dict['x_rotation'], weight= default_plot_dict['weight'], fontfamily=default_plot_dict['family'])
    if 'yticks' in default_plot_dict.keys():
        _= axs.set_yticks(default_plot_dict['yticks'])
    if 'yticklabels' in default_plot_dict.keys():
        _= axs.set_yticklabels(default_plot_dict['yticklabels'], fontsize= default_plot_dict['fontsize'], rotation= default_plot_dict['y_rotation'], weight= default_plot_dict['weight'], fontfamily=default_plot_dict['family'])
    if 'xlabel' in default_plot_dict.keys():
        _= axs.set_xlabel(default_plot_dict['xlabel'], fontsize= default_plot_dict['fontsize'], labelpad= default_plot_dict['pad'], weight= default_plot_dict['weight'], fontfamily= default_plot_dict['family'])
    if 'ylabel' in default_plot_dict.keys():
        _= axs.set_ylabel(default_plot_dict['ylabel'], fontsize= default_plot_dict['fontsize'], labelpad= default_plot_dict['pad'], weight= default_plot_dict['weight'], fontfamily= default_plot_dict['family'])
    if 'title' in default_plot_dict.keys():
        _= axs.set_title(default_plot_dict['title'], fontsize= default_plot_dict['fontsize'], pad= default_plot_dict['pad']*4, weight= 'bold', fontfamily= default_plot_dict['family'])
    if 'logy' in default_plot_dict.keys():
        _= axs.set_yscale('log', base=10)
    if 'logx' in default_plot_dict.keys():
        _= axs.set_xscale('log', base=10)

    if default_plot_dict['subplot_behavior']:
        try:
            _= axs.label_outer()
        except Exception as e:
            pass

    if default_plot_dict['legend'] and default_plot_dict['fixlegend']:
            
        axs.legend(handletextpad= 0.5, loc= default_plot_dict['legend_loc'])
        
        if default_plot_dict['rm_legend_title']:
            axs.legend().set_title('')
            
        if default_plot_dict['snsfig']:
            plt.setp(axs.get_legend().get_texts(),fontsize= default_plot_dict['fontsize']/2)
            plt.setp(axs.get_legend().get_title(),fontsize= default_plot_dict['fontsize']/2)
            plt.setp(axs.get_legend().get_frame(),visible=False)
            [plt.setp(x, linewidth=1) for x in axs.get_legend().findobj() if isinstance(x,matplotlib.lines.Line2D)]
            [plt.setp(x, width=3, height=3) for x in axs.get_legend().findobj() if isinstance(x,matplotlib.patches.Rectangle)]
            [plt.setp(x, width=0.1, height=0.1) for x in axs.get_legend().findobj() if isinstance(x,matplotlib.offsetbox.DrawingArea)]
            [plt.setp(x, sizes=[1]) for x in axs.get_legend().findobj() if isinstance(x,matplotlib.collections.PathCollection)]
        else:
            axs.legend(fontsize= default_plot_dict['fontsize']/2, markerscale= default_plot_dict['markerscale'], frameon=False)
                
    elif not default_plot_dict['legend']:
        axs.get_legend().remove()

    return axs


def _index_to_label(index):
    """Convert a pandas index or multiindex to an axis label."""
    if isinstance(index, pd.MultiIndex):
        return "-".join(map(to_utf8, index.names))
    else:
        return index.name

def _index_to_ticklabels(index):
    """Convert a pandas index or multiindex into ticklabels."""
    if isinstance(index, pd.MultiIndex):
        return ["-".join(map(to_utf8, i)) for i in index.values]
    else:
        return index.values

def _matrix_mask(data, mask):
    """Ensure that data and mask are compatabile and add missing values.

    Values will be plotted for cells where ``mask`` is ``False``.

    ``data`` is expected to be a DataFrame; ``mask`` can be an array or
    a DataFrame.

    """
    if mask is None:
        mask = np.zeros(data.shape, np.bool)

    if isinstance(mask, np.ndarray):
        # For array masks, ensure that shape matches data then convert
        if mask.shape != data.shape:
            raise ValueError("Mask must have the same shape as data.")

        mask = pd.DataFrame(mask,
                            index=data.index,
                            columns=data.columns,
                            dtype=np.bool)

    elif isinstance(mask, pd.DataFrame):
        # For DataFrame masks, ensure that semantic labels match data
        if not mask.index.equals(data.index) \
           and mask.columns.equals(data.columns):
            err = "Mask must have the same index and columns as data."
            raise ValueError(err)

    # Add any cells with missing data to the mask
    # This works around an issue where `plt.pcolormesh` doesn't represent
    # missing data properly
    mask = mask | pd.isnull(data)

    return mask


class _HeatMapper2(object):
    """Draw a heatmap plot of a matrix with nice labels and colormaps."""

    def __init__(self, data, vmin, vmax, cmap, center, robust, annot, fmt,
                 annot_kws, cellsize, cellsize_vmax,
                 cbar, cbar_kws,
                 xticklabels=True, yticklabels=True, mask=None, ax_kws=None, rect_kws=None, fontsize=4):
        """Initialize the plotting object."""
        # We always want to have a DataFrame with semantic information
        # and an ndarray to pass to matplotlib
        if isinstance(data, pd.DataFrame):
            plot_data = data.values
        else:
            plot_data = np.asarray(data)
            data = pd.DataFrame(plot_data)

        # Validate the mask and convet to DataFrame
        mask = _matrix_mask(data, mask)

        plot_data = np.ma.masked_where(np.asarray(mask), plot_data)

        # Get good names for the rows and columns
        xtickevery = 1
        if isinstance(xticklabels, int):
            xtickevery = xticklabels
            xticklabels = _index_to_ticklabels(data.columns)
        elif xticklabels is True:
            xticklabels = _index_to_ticklabels(data.columns)
        elif xticklabels is False:
            xticklabels = []

        ytickevery = 1
        if isinstance(yticklabels, int):
            ytickevery = yticklabels
            yticklabels = _index_to_ticklabels(data.index)
        elif yticklabels is True:
            yticklabels = _index_to_ticklabels(data.index)
        elif yticklabels is False:
            yticklabels = []

        # Get the positions and used label for the ticks
        nx, ny = data.T.shape

        if not len(xticklabels):
            self.xticks = []
            self.xticklabels = []
        elif isinstance(xticklabels, str) and xticklabels == "auto":
            self.xticks = "auto"
            self.xticklabels = _index_to_ticklabels(data.columns)
        else:
            self.xticks, self.xticklabels = self._skip_ticks(xticklabels,
                                                             xtickevery)

        if not len(yticklabels):
            self.yticks = []
            self.yticklabels = []
        elif isinstance(yticklabels, str) and yticklabels == "auto":
            self.yticks = "auto"
            self.yticklabels = _index_to_ticklabels(data.index)
        else:
            self.yticks, self.yticklabels = self._skip_ticks(yticklabels,
                                                             ytickevery)

        # Get good names for the axis labels
        xlabel = _index_to_label(data.columns)
        ylabel = _index_to_label(data.index)
        self.xlabel = xlabel if xlabel is not None else ""
        self.ylabel = ylabel if ylabel is not None else ""

        # Determine good default values for the colormapping
        self._determine_cmap_params(plot_data, vmin, vmax,
                                    cmap, center, robust)

        # Determine good default values for cell size
        self._determine_cellsize_params(plot_data, cellsize, cellsize_vmax)

        # Sort out the annotations
        if annot is None:
            annot = False
            annot_data = None
        elif isinstance(annot, bool):
            if annot:
                annot_data = plot_data
            else:
                annot_data = None
        else:
            try:
                annot_data = annot.values
            except AttributeError:
                annot_data = annot
            if annot.shape != plot_data.shape:
                raise ValueError('Data supplied to "annot" must be the same '
                                 'shape as the data to plot.')
            annot = True

        # Save other attributes to the object
        self.data = data
        self.plot_data = plot_data

        self.annot = annot
        self.annot_data = annot_data

        self.fmt = fmt
        self.annot_kws = {} if annot_kws is None else annot_kws
        self.annot_kws.setdefault('color', "black")
        self.annot_kws.setdefault('ha', "center")
        self.annot_kws.setdefault('va', "center")
        self.annot_kws.setdefault('fontsize', fontsize)
        self.cbar = cbar
        self.cbar_kws = {} if cbar_kws is None else cbar_kws
        self.cbar_kws.setdefault('ticks', matplotlib.ticker.MaxNLocator(6))
        self.ax_kws = {} if ax_kws is None else ax_kws
        self.rect_kws = {} if rect_kws is None else rect_kws
        # self.rect_kws.setdefault('edgecolor', "black")

    def _determine_cmap_params(self, plot_data, vmin, vmax,
                               cmap, center, robust):
        """Use some heuristics to set good defaults for colorbar and range."""
        calc_data = plot_data.data[~np.isnan(plot_data.data)]
        if vmin is None:
            vmin = np.percentile(calc_data, 2) if robust else calc_data.min()
        if vmax is None:
            vmax = np.percentile(calc_data, 98) if robust else calc_data.max()
        self.vmin, self.vmax = vmin, vmax

        # Choose default colormaps if not provided
        if cmap is None:
            if center is None:
                self.cmap = cm.rocket
            else:
                self.cmap = cm.icefire
        elif isinstance(cmap, str):
            self.cmap = matplotlib.cm.get_cmap(cmap)
        elif isinstance(cmap, list):
            self.cmap = matplotlib.colors.ListedColormap(cmap)
        else:
            self.cmap = cmap

        # Recenter a divergent colormap
        if center is not None:
            vrange = max(vmax - center, center - vmin)
            normlize = matplotlib.colors.Normalize(center - vrange, center + vrange)
            cmin, cmax = normlize([vmin, vmax])
            cc = np.linspace(cmin, cmax, 256)
            self.cmap = matplotlib.colors.ListedColormap(self.cmap(cc))

    def _determine_cellsize_params(self, plot_data, cellsize, cellsize_vmax):

        if cellsize is None:
            self.cellsize = np.ones(plot_data.shape)
            self.cellsize_vmax = 1.0
        else:
            if isinstance(cellsize, pd.DataFrame):
                cellsize = cellsize.values
            self.cellsize = cellsize
            if cellsize_vmax is None:
                cellsize_vmax = cellsize.max()
            self.cellsize_vmax = cellsize_vmax

    def _skip_ticks(self, labels, tickevery):
        """Return ticks and labels at evenly spaced intervals."""
        n = len(labels)
        if tickevery == 0:
            ticks, labels = [], []
        elif tickevery == 1:
            ticks, labels = np.arange(n) + .5, labels
        else:
            start, end, step = 0, n, tickevery
            ticks = np.arange(start, end, step) + .5
            labels = labels[start:end:step]
        return ticks, labels

    def _auto_ticks(self, ax, labels, axis, fontsize):
        """Determine ticks and ticklabels that minimize overlap."""
        transform = ax.figure.dpi_scale_trans.inverted()
        bbox = ax.get_window_extent().transformed(transform)
        size = [bbox.width, bbox.height][axis]
        axis = [ax.xaxis, ax.yaxis][axis]
        tick, = axis.set_ticks([0])
        max_ticks = int(size // (fontsize / 72))
        if max_ticks < 1:
            return [], []
        tick_every = len(labels) // max_ticks + 1
        tick_every = 1 if tick_every == 0 else tick_every
        ticks, labels = self._skip_ticks(labels, tick_every)
        return ticks, labels

    def plot(self, ax, cax, fontsize, rowcolors=None, colcolors=None, ref_sizes=None, ref_labels=None):
        """Draw the heatmap on the provided Axes."""

        # Remove all the Axes spines
        despine(ax=ax, left=True, bottom=True)

        # Draw the heatmap and annotate
        height, width = self.plot_data.shape
        xpos, ypos = np.meshgrid(np.arange(width) + .5, np.arange(height) + .5)

        data = self.plot_data.data
        cellsize = self.cellsize

        mask = self.plot_data.mask
        if not isinstance(mask, np.ndarray) and not mask:
            mask = np.zeros(self.plot_data.shape, np.bool)

        annot_data = self.annot_data
        if not self.annot:
            annot_data = np.zeros(self.plot_data.shape)

        # Draw rectangles instead of using pcolormesh
        # Might be slower than original heatmap
        for x, y, m, val, s, an_val in zip(xpos.flat, ypos.flat, mask.flat, data.flat, cellsize.flat, annot_data.flat):
            if not m:
                vv = (val - self.vmin) / (self.vmax - self.vmin)
                size = np.clip(s / self.cellsize_vmax, 0.1, 1.0)
                color = self.cmap(vv)
                rect = plt.Rectangle([x - size / 2, y - size / 2], size, size, facecolor=color, label=None, **self.rect_kws)
                ax.add_patch(rect)

                if self.annot:
                    annotation = ("{:" + self.fmt + "}").format(an_val)
                    text = ax.text(x, y, annotation, **self.annot_kws)
                    # add edge to text
                    text_luminance = relative_luminance(text.get_color())
                    text_edge_color = ".15" if text_luminance > .408 else "w"
                    text.set_path_effects([matplotlib.patheffects.withStroke(linewidth=1, foreground=text_edge_color)])
        
        ## Draw rectangles for size scale using specific reference sizes

        if ref_sizes is not None:

            # ref_s = [1.30,2.00,3.00,5.00,10.00,self.cellsize_vmax]
            # ref_l = ['0.05','0.01','1e-3','1e-5','1e-10','maxsize: '+'{:.1e}'.format(10**(-1*self.cellsize_vmax))]

            ref_s = ref_sizes + [self.cellsize_vmax]
            ref_l = ref_labels + ['maxsize']
            ref_x = -10*np.ones(len(ref_s))
            ref_y = np.arange(len(ref_s))
            
            for x,y,s,l in zip(ref_x,ref_y,ref_s,ref_l):
                size = np.clip(s / self.cellsize_vmax, 0.1, 1.0)
                rect = plt.Rectangle([x - size / 2, y - size / 2], size, size, facecolor='k', label=l, linewidth=0, edgecolor=None, **self.rect_kws)
                ax.add_patch(rect)
                ax.text(x, y, l, **self.annot_kws)
        
        ## Draw rectangles to provide a row color annotation 
        if rowcolors is not None:
            for i,r in enumerate(rowcolors):
                for x,y,c in zip(xpos[:,0]-(15+i),ypos[:,0],r):
                    size = 1
                    rect = plt.Rectangle([x - size / 2, y - size / 2], size, size, facecolor=c, label=None, linewidth=0, edgecolor=None, **self.rect_kws)
                    ax.add_patch(rect)
            
        ## Draw rectangles to provide a column color annotation  
        if colcolors is not None:
            for i,c in enumerate(colcolors):
                for x,y,c in zip(xpos[0,:],ypos[0,:]-(10+i),c):
                    size = 1
                    rect = plt.Rectangle([x - size / 2, y - size / 2], size, size, facecolor=c, label=None, linewidth=0, edgecolor=None, **self.rect_kws)
                    ax.add_patch(rect)

        # Set the axis limits
        ax.set(xlim=(0, self.data.shape[1]), ylim=(0, self.data.shape[0]))

        # Set other attributes
        ax.set(**self.ax_kws)

        if self.cbar:
            norm = matplotlib.colors.Normalize(vmin=self.vmin, vmax=self.vmax)
            scalar_mappable = matplotlib.cm.ScalarMappable(cmap=self.cmap, norm=norm)
            scalar_mappable.set_array(self.plot_data.data)
            cb = ax.figure.colorbar(scalar_mappable, cax, ax, **self.cbar_kws)
            cb.outline.set_linewidth(0)
            cb.ax.tick_params(labelsize=fontsize) 

        # Add row and column labels
        if isinstance(self.xticks, str) and self.xticks == "auto":
            xticks, xticklabels = self._auto_ticks(ax, self.xticklabels, axis=0, fontsize=fontsize)
        else:
            xticks, xticklabels = self.xticks, self.xticklabels

        if isinstance(self.yticks, str) and self.yticks == "auto":
            yticks, yticklabels = self._auto_ticks(ax, self.yticklabels, axis=1, fontsize=fontsize)
        else:
            yticks, yticklabels = self.yticks, self.yticklabels

        ax.set(xticks=xticks, yticks=yticks)
        xtl = ax.set_xticklabels(xticklabels, fontsize=fontsize)
        ytl = ax.set_yticklabels(yticklabels, rotation="vertical", fontsize=fontsize)

        # Possibly rotate them if they overlap
        ax.figure.draw(ax.figure.canvas.get_renderer())
        if axis_ticklabels_overlap(xtl):
            plt.setp(xtl, rotation="vertical")
        if axis_ticklabels_overlap(ytl):
            plt.setp(ytl, rotation="horizontal")

        # Add the axis labels
        ax.set(xlabel=self.xlabel, ylabel=self.ylabel)

        # Invert the y axis to show the plot in matrix form
        ax.invert_yaxis()


def heatmap2(data, vmin=None, vmax=None, cmap=None, center=None, robust=False,
            annot=None, fmt=".2g", annot_kws=None,
            cellsize=None, cellsize_vmax=None,
            ref_sizes=None, ref_labels=None,
            cbar=True, cbar_kws=None, cbar_ax=None,
            square=True, xticklabels="auto", yticklabels="auto",rowcolors=None,colcolors=None,
            mask=None, ax=None, ax_kws=None, rect_kws=None, fontsize=4, figsize=(2,2)):

    # Initialize the plotter object
    plotter = _HeatMapper2(data, vmin, vmax, cmap, center, robust,
                          annot, fmt, annot_kws,
                          cellsize, cellsize_vmax,
                          cbar, cbar_kws, xticklabels,
                          yticklabels, mask, ax_kws, rect_kws, fontsize)

    # Draw the plot and return the Axes
    if ax is None:
        fig,ax = plt.subplots(figsize=figsize)
    if square:
        ax.set_aspect("equal")

    # delete grid
    ax.grid(False)
    
    plotter.plot(ax, cbar_ax, fontsize=fontsize, rowcolors=rowcolors, colcolors=colcolors, ref_sizes=ref_sizes, ref_labels=ref_labels)
      
    return ax