import os
import re
import math
import scipy
import random
import sklearn
import anndata
import itertools
import matplotlib
import numpy as np
import scanpy as sc
import pandas as pd
import seaborn as sns
from glob import glob
from statannot import add_stat_annotation

import matplotlib.gridspec as gridspec
import matplotlib.patheffects as patheffects
from matplotlib import pyplot as plt
from matplotlib.legend import Legend
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

from seaborn import cm
from seaborn.axisgrid import Grid
from seaborn.utils import (despine, axis_ticklabels_overlap, relative_luminance, to_utf8)
from seaborn.axisgrid import Grid
from seaborn.utils import (despine, axis_ticklabels_overlap, relative_luminance, to_utf8)

from kgbio.scseq import qcfunctions as qc


def fix_plot(axs,fontsize=4,pad=0.4,grid=False,xlabel=None,ylabel=None,xticks=None,yticks=None,
              xlim=None,ylim=None,xticklabels=None,yticklabels=None,x_rotation=0,y_rotation=0,
             snsfig=True,legend=True,fixlegend=False,markerscale=0.2,title=None,weight='regular'):

    # set font
    plt.rcParams['font.family'] = 'Roboto'
    plt.rcParams['font.weight'] = 'regular'

    _= axs.tick_params(axis='both',labelsize=fontsize,pad=pad)
    _= axs.grid(visible=grid)

    if xlim is not None:
        _= axs.set_xlim(xlim)
    if ylim is not None:
        _= axs.set_ylim(ylim)
    if xticks is not None:
        _= axs.set_xticks(xticks)
    if xticklabels is not None:
        _= axs.set_xticklabels(xticklabels,fontsize=fontsize,rotation=x_rotation,weight=weight,fontfamily='Roboto')
    if yticks is not None:
        _= axs.set_yticks(yticks)
    if yticklabels is not None:
        _= axs.set_yticklabels(yticklabels,fontsize=fontsize,rotation=y_rotation,weight=weight,fontfamily='Roboto')
    if xlabel is not None:
        _= axs.set_xlabel(xlabel,fontsize=fontsize,labelpad=pad,weight=weight,fontfamily='Roboto')
    if ylabel is not None:
        _= axs.set_ylabel(ylabel,fontsize=fontsize,labelpad=pad,weight=weight,fontfamily='Roboto')
    if title is not None:
        _= axs.set_title(title,fontsize=fontsize,pad=pad*4,weight='bold',fontfamily='Roboto')

    if legend and fixlegend:    
        if snsfig:
            plt.setp(axs.get_legend().get_texts(),fontsize=fontsize/2)
            plt.setp(axs.get_legend().get_title(),fontsize=fontsize/2)
            plt.setp(axs.get_legend().get_frame(),visible=False)
            [plt.setp(x,linewidth=1,markersize=0.1) for x in axs.get_legend().findobj() if isinstance(x,matplotlib.lines.Line2D)]
            [plt.setp(x,width=3,height=3) for x in axs.get_legend().findobj() if isinstance(x,matplotlib.patches.Rectangle)]
            [plt.setp(x,width=0.1,height=0.1) for x in axs.get_legend().findobj() if isinstance(x,matplotlib.offsetbox.DrawingArea)]
            [plt.setp(x,sizes=[1]) for x in axs.get_legend().findobj() if isinstance(x,matplotlib.collections.PathCollection)]
        else:
            axs.legend(fontsize=fontsize/2,markerscale=markerscale,frameon=False)
    elif not legend:
        axs.get_legend().remove()

    return axs


def read_in_h5(inpath=None, samplename=None, modality=None, image_path=None, species=None, gene_annot=None):

    if ('spatial' in modality):
        adata = sc.read_visium(inpath+'/outs/', count_file='filtered_feature_bc_matrix.h5', source_image_path=image_path)
    else:
        adata = sc.read_10x_h5(inpath+'/outs/filtered_feature_bc_matrix.h5')

    adata.obs.index = [samplename+'_'+x.split('-')[0] for x in adata.obs.index]
    adata.obs['sampleID'] = samplename

    adata.var_names_make_unique()

    adata.obs['sparcity'] = qc.calc_sparcity(adata)    

    if species=='hsapiens':
        adata.var['mt'] = adata.var_names.str.startswith('MT-')
    else:
        adata.var['mt'] = adata.var_names.str.startswith('mt-')

    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], inplace=True)

    adata = qc.annotate_genes(adata, remove_noncoding=False, species=species)

    return adata


def run_dimreduc_clustering(adata, ntopgenes=2000, hvg_flavor='seurat_v3', regress_var=None, resolution=0.8, nneighbors=30, npcs=40, do_normalize=True, do_scale=False, do_regress=False, do_umap=True, do_tsne=False, n_jobs=1):

    if (hvg_flavor=='seurat_v3'):

        sc.pp.highly_variable_genes(adata, n_top_genes=ntopgenes, flavor=hvg_flavor, subset=False, inplace=True)

        if do_normalize:
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)

    else:

        if do_normalize:
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)

        sc.pp.highly_variable_genes(adata, n_top_genes=ntopgenes, flavor=hvg_flavor, subset=False, inplace=True)


    if do_regress:
        sc.pp.regress_out(adata, keys=regress_var, n_jobs=n_jobs)

    if do_scale:
        sc.pp.scale(adata, max_value=10)


    sc.tl.pca(adata, use_highly_variable=True)

    sc.pp.neighbors(adata, n_neighbors=nneighbors, n_pcs=npcs)

    if do_umap:
        sc.tl.umap(adata)
    if do_tsne:
        sc.tl.tsne(adata, n_pcs=npcs, njobs=n_jobs)

    sc.tl.leiden(adata, resolution=resolution)
    

    return adata


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



def umap_density(adata=None, df=None, embedding='X_umap', t=0.2, lv=5, gsize=200, alpha=0.4, colors=None, groupby=None, hue=None, fill=True, include_scatter=True, dotsize=0.005, figsize=None):

    if adata is not None:
        df = pd.DataFrame(adata.obsm[embedding], columns=['emb1','emb2'], index=adata.obs.index)
        df = df.merge(adata.obs, left_index=True, right_index=True)

    st = df[groupby].unique().tolist()

    if colors is None:
        
        if (len(st)<=10):
            test = plt.get_cmap('tab10')
        elif (len(st)<=20):
            test = plt.get_cmap('tab20')
        else:
            print('please provide colors because more than 20 categories')
            return

        colors = {g:test(i) for i,g in enumerate(st)}

    if figsize is None:
        figsize = (len(st)*2,2)
        
    fig,axs = plt.subplots(nrows=1,
                           ncols=len(st),
                           sharex=True,
                           sharey=True,
                           figsize=figsize)

    for s,ax in zip(st,axs.flat):
        
        tmp = df.loc[(df[groupby]==s),:]
        
        _= sns.kdeplot(x='emb1',
                       y='emb2',
                       data=tmp,
                       hue=hue,
                       fill=fill,
                       palette=colors,
                       thresh=t,
                       alpha=alpha,
                       levels=lv,
                       gridsize=gsize,
                       ax=ax)

        # if hue is None:
        #     _= sns.kdeplot(x='emb1',y='emb2',data=tmp,fill=fill,color=colors[s],thresh=t,alpha=alpha,levels=lv,gridsize=gsize,ax=ax,legend=False)
        # else:
        #     _= sns.kdeplot(x='emb1',y='emb2',data=tmp,hue=hue,fill=fill,palette=colors,thresh=t,alpha=alpha,levels=lv,gridsize=gsize,ax=ax,legend=True)

        if include_scatter:
            _= ax.scatter(x=df['emb1'],
                          y=df['emb2'],
                          c='tab:gray',
                          s=dotsize,
                          marker='.',
                          linewidths=0)
        
        _= fix_plot(ax,
                    title=s,
                    fontsize=4,
                    pad=2,
                    fixlegend=True)

    return fig,axs


def plot_gex(adata, GOI=None, use_obs=False, dgex=None, groupby='leiden', use_raw=False, use_FDR=True, dendro=False, plot_type='dotplot', cmap='Reds', figsize=None, vmin=0, vmax=1, fontsize=4, size_max=None):

    """
    plot type options are: dotplot, matrixplot, scanpy_heatmap, custom_heatmap

    custom_heatmap returns effect_size, pval, fig, axs
    all others return fig

    """

    if use_obs:
        tmp = adata.copy()
    else:
        tmp = adata[:,GOI].copy()

    grp = sorted(tmp.obs[groupby].unique().tolist())

    if dendro:
        sc.tl.dendrogram(tmp, groupby=groupby, var_names=GOI, use_raw=use_raw, optimal_ordering=True)    
    
    sns.set_style("white", rc={"font.family":"Helvetica","axes.grid":False})                                                  
    sns.set_context("paper", rc={"font.size":fontsize,"axes.titlesize":fontsize,"axes.labelsize":fontsize,"font.family":"Helvetica","xtick.labelsize":fontsize,"ytick.labelsize":fontsize})
    
    if figsize is None:
        figsize = (2,2)

    if (plot_type=='dotplot'):
        
        test = sc.pl.dotplot(tmp,var_names=GOI,groupby=groupby,use_raw=use_raw,standard_scale='var',vmin=vmin,vmax=vmax,
                             dendrogram=dendro,swap_axes=True,figsize=figsize,show=False,return_fig=True)
        
        test.style(color_on='square',cmap=cmap,dot_edge_color='white',
                   dot_edge_lw=0.5,grid=True,size_exponent=3,largest_dot=1,
                   dot_min=0,dot_max=1)
        
        return test
    
    elif (plot_type=='matrixplot'):
        
        test = sc.pl.matrixplot(tmp,var_names=GOI,groupby=groupby,use_raw=use_raw,standard_scale='var',vmin=vmin,vmax=vmax,
                                dendrogram=dendro,swap_axes=True,cmap=cmap,figsize=figsize,show=False,return_fig=True)
        
        test.style(edge_lw=0)
        
        return test
    
    elif (plot_type=='scanpy_heatmap'):
        
        ax = sc.pl.heatmap(tmp,var_names=GOI,groupby=groupby,use_raw=use_raw,standard_scale='var',vmin=vmin,vmax=vmax,
                           dendrogram=dendro,cmap=cmap,swap_axes=True,show_gene_labels=True,figsize=figsize,show=False)
    
        return ax

    elif (plot_type=='custom_heatmap'):
        
        effect_size = pd.DataFrame(index=GOI, columns=grp, dtype=np.float64)
        pval = pd.DataFrame(index=GOI, columns=grp, dtype=np.float64)

        for idx in effect_size.index:
            for col in effect_size.columns:

                effect_size.loc[idx,col] = dgex.loc[(dgex['names']==idx)&(dgex['group']==int(col)),'logfoldchanges'].item()

                if use_FDR:
                    pval.loc[idx,col] = dgex.loc[(dgex['names']==idx)&(dgex['group']==int(col)),'pvals_adj'].item()
                else:
                    pval.loc[idx,col] = dgex.loc[(dgex['names']==idx)&(dgex['group']==int(col)),'pvals'].item()

        pval = -1*np.log10(pval.to_numpy(dtype=np.float64))
        pval[np.isinf(pval)] = np.max(pval[~np.isinf(pval)])
        pval = pd.DataFrame(pval,index=GOI,columns=grp)
        
        d = scipy.spatial.distance.pdist(effect_size.to_numpy(dtype='float64'), metric='euclidean')
        l = scipy.cluster.hierarchy.linkage(d, metric='euclidean', method='complete', optimal_ordering=True)
        dn = scipy.cluster.hierarchy.dendrogram(l, no_plot=True)
        order = dn['leaves']
        
        effect_size = effect_size.iloc[order,:]
        pval = pval.iloc[order,:]
        
        if size_max is None:
            size_max = np.amax(pval.to_numpy())

        ref_s = [1.30,size_max]
        ref_l = ['0.05','maxsize: '+'{:.1e}'.format(10**(-1*size_max))]


        fig,axs = plt.subplots(nrows=1,
                               ncols=2,
                               figsize=figsize)
        
        heatmap2(effect_size,
                 cmap='RdBu_r',
                 vmin=vmin,
                 vmax=vmax,
                 cellsize=pval,
                 square=True,
                 cellsize_vmax=size_max,
                 ref_sizes=ref_s,
                 ref_labels=ref_l,
                 fontsize=fontsize,
                 figsize=figsize,
                 ax=axs[0])

        icoord = np.array(dn['icoord'] )
        dcoord = np.array(dn['dcoord'] )

        for xs, ys in zip(dcoord, icoord):
            _= axs[1].plot(xs,
                           ys,
                           color='k',
                           linewidth=0.5)

        _= fix_plot(axs[1],
                    xticks=[],
                    yticks=[])

        return effect_size, pval, fig, axs


def gex_clustermap(adata=None,GOI=None,groupby=None,use_raw=False,
                   genes_on='rows',
                   row_cluster=True,col_cluster=True,
                   row_order=None,col_order=None,
                   standard_scale=False,zscore=True,
                   dendrogram_ratio=None,
                   cbar=True,cbar_shrink=1.0,cbar_title='',
                   xlabel=None,ylabel=None,x_rotation=0,
                   vmin=-3,vmax=3,figsize=None,fontsize=2):
    
    if GOI is None:
        GOI = adata.var_names
    if figsize is None:
        figsize = (2,3)
    if dendrogram_ratio is None:
        dendrogram_ratio = (0.3,0.05)
    if xlabel is None:
        xlabel = groupby
    if ylabel is None:
        ylabel = ''
    
    if use_raw:
        df = pd.DataFrame(adata[:,GOI].raw.X.toarray(),index=adata.obs.index,columns=GOI)
    else:
        df = pd.DataFrame(adata[:,GOI].X.toarray(),index=adata.obs.index,columns=GOI)
        
    df = df.merge(adata.obs,how='left',left_index=True,right_index=True)
    df = df.groupby(groupby).mean()[GOI]
    
    norm_dim = 1
    if genes_on=='rows':
        df = df.T
        norm_dim = 0
        
    if row_order is not None:
        df = df.loc[row_order,:].copy()
    if col_order is not None:
        df = df.loc[:,col_order].copy()
    
    if not cbar:
        cbar_pos = None
    else:
        cbar_pos = (-0.02, 0.92, 0.05*cbar_shrink, 0.18*cbar_shrink)
        
    if standard_scale:
        g = sns.clustermap(df,
                           method='complete',
                           standard_scale=norm_dim,
                           row_cluster=row_cluster,
                           col_cluster=col_cluster,
                           dendrogram_ratio=dendrogram_ratio,
                           cmap='Reds',
                           vmin=vmin,vmax=vmax,
                           cbar_kws=dict(ticks=[vmin, 0.50, vmax], orientation='horizontal'),
                           cbar_pos=cbar_pos,
                           xticklabels=df.columns,
                           yticklabels=df.index,
                           figsize=figsize)
    elif zscore:
        g = sns.clustermap(df,
                           method='complete',
                           z_score=norm_dim,
                           row_cluster=row_cluster,
                           col_cluster=col_cluster,
                           dendrogram_ratio=dendrogram_ratio,
                           cmap='RdBu_r',
                           vmin=vmin,vmax=vmax,
                           cbar_kws=dict(ticks=[vmin, 0, vmax], orientation='horizontal'),
                           cbar_pos=cbar_pos,
                           xticklabels=df.columns,
                           yticklabels=df.index,
                           figsize=figsize)
    
    g.ax_heatmap.set_yticks(np.arange(df.shape[0])+0.5)
    g.ax_heatmap.set_xticks(np.arange(df.shape[1])+0.5)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=fontsize)
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), fontsize=fontsize/2, rotation=x_rotation)
    g.ax_heatmap.set_xlabel(xlabel, fontsize=fontsize)
    g.ax_heatmap.set_ylabel(ylabel, fontsize=fontsize)
    g.ax_heatmap.tick_params(axis='both', length=1, width=0.5)
    
    if cbar:
        x0, y0, _w, _h = g.cbar_pos
        g.ax_cbar.set_position([x0, y0, g.ax_row_dendrogram.get_position().width*cbar_shrink, 0.02*cbar_shrink])
        g.ax_cbar.set_title(cbar_title, fontdict={'fontsize': fontsize/2})
        g.ax_cbar.tick_params(axis='x', labelsize=fontsize/4, length=0.5, width=0.5)
    
    return g


