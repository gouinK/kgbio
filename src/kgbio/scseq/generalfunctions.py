import os
import re
import math
import scvi
import scipy
import random
import sklearn
import anndata
import itertools
import celltypist
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
from matplotlib import font_manager

from seaborn import cm
from seaborn.axisgrid import Grid
from seaborn.utils import (despine, axis_ticklabels_overlap, relative_luminance, to_utf8)
from seaborn.axisgrid import Grid
from seaborn.utils import (despine, axis_ticklabels_overlap, relative_luminance, to_utf8)

from kgbio.scseq import qcfunctions as qc
from kgbio.utils.plotting import fix_plot
from kgbio.utils.plotting import heatmap2


def read_in_h5(inpath=None, samplename=None, modality=None, image_path=None, species=None):

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


def run_dimreduc_clustering(adata=None, ntopgenes=2000, hvg_flavor='seurat_v3', regress_var=None, resolution=0.8, nneighbors=30, npcs=40, do_normalize=True, do_scale=False, do_regress=False, do_umap=True, do_tsne=False, n_jobs=1):

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
        
        plot_dict = {'title': s, 'fontsize': 4, 'pad': 2, 'fixlegend': True}
        _= fix_plot(ax, plot_dict=plot_dict)

    return fig, axs


def plot_gex(adata=None, GOI=None, use_obs=False, dgex=None, groupby='leiden', use_raw=False, use_FDR=True, dendro=False, plot_type='dotplot', embedding=None, dotsize=8, cmap='Reds', figsize=None, vmin=0, vmax=1, fontsize=4, size_max=None):

    """
    plot type options are: dotplot, matrixplot, scanpy_heatmap, custom_heatmap, embedding

    custom_heatmap returns effect_size, pval, fig, axs
    embedding returns fig, axs
    all others return fig

    """

    if (plot_type == 'embedding'):
        
        numGenes = len(GOI)
        ncols = 4
        nrows = int(np.ceil(numGenes / ncols))
        
        if figsize is None:
            figsize = (ncols, nrows)
            
        fig, axs = plt.subplots(nrows= nrows,
                                ncols= ncols,
                                sharex= True,
                                sharey= True,
                                gridspec_kw= {'hspace': 0.3, 'wspace': 0.1},
                                figsize= figsize)
        
        for g, ax in zip(GOI, axs.flat):
            
            _= sc.pl.embedding(adata,
                                basis= embedding,
                                color= g,
                                use_raw= use_raw,
                                color_map= cmap,
                                size= dotsize,
                                frameon= False,
                                add_outline= True,
                                colorbar_loc= None,
                                legend_fontsize= (fontsize/2),
                                show= False,
                                ax= ax)
            
            ## Make custom colorbar
            if use_raw:
                arr = adata[:,g].raw.X
            else:
                arr = adata[:,g].X
                
            norm = matplotlib.colors.Normalize(vmin= np.min((0, np.min(arr))), 
                                               vmax= np.max(arr))
            sm = matplotlib.cm.ScalarMappable(cmap= cmap, norm= norm) 
            cax = plt.colorbar(mappable=sm, ax=ax, shrink=0.5)
            cax.set_label(label='expression', size=(fontsize/2), labelpad=1)
            cax.ax.tick_params(labelsize= (fontsize/2), pad=1, length=0.5, width=0.5)
            
            ## Update plot params
            plot_dict = {'xlabel': f'{embedding}_1',
                        'ylabel': f'{embedding}_2',
                        'title': g,
                        'fontsize': fontsize} 
            
            _= fix_plot(ax, plot_dict=plot_dict)
            
        return fig, axs

    else:

        if use_obs:
            tmp = adata.copy()
        else:
            tmp = adata[:,GOI].copy()

        grp = sorted(tmp.obs[groupby].unique().tolist())

        if dendro:
            sc.tl.dendrogram(tmp, groupby=groupby, var_names=GOI, use_raw=use_raw, optimal_ordering=True)    
        
        if figsize is None:
            figsize = (2,2)

        if (plot_type == 'dotplot'):
            
            test = sc.pl.dotplot(tmp,var_names=GOI,groupby=groupby,use_raw=use_raw,standard_scale='var',vmin=vmin,vmax=vmax,
                                dendrogram=dendro,swap_axes=True,figsize=figsize,show=False,return_fig=True)
            
            test.style(color_on='square',cmap=cmap,dot_edge_color='white',
                    dot_edge_lw=0.5,grid=True,size_exponent=3,largest_dot=1,
                    dot_min=0,dot_max=1)
            
            return test
        
        elif (plot_type == 'matrixplot'):
            
            test = sc.pl.matrixplot(tmp,var_names=GOI,groupby=groupby,use_raw=use_raw,standard_scale='var',vmin=vmin,vmax=vmax,
                                    dendrogram=dendro,swap_axes=True,cmap=cmap,figsize=figsize,show=False,return_fig=True)
            
            test.style(edge_lw=0)
            
            return test
        
        elif (plot_type == 'scanpy_heatmap'):
            
            ax = sc.pl.heatmap(tmp,var_names=GOI,groupby=groupby,use_raw=use_raw,standard_scale='var',vmin=vmin,vmax=vmax,
                            dendrogram=dendro,cmap=cmap,swap_axes=True,show_gene_labels=True,figsize=figsize,show=False)
        
            return ax

        elif (plot_type == 'custom_heatmap'):
            
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

            plot_dict = {'xticks': [], 'yticks': []}
            _= fix_plot(axs[1], plot_dict=plot_dict)

            return effect_size, pval, fig, axs


def gex_clustermap(adata=None, GOI=None, groupby=None, use_raw=False,
                   genes_on='rows',
                   row_cluster=True, col_cluster=True,
                   row_order=None, col_order=None,
                   standard_scale=False, zscore=True,
                   dendrogram_ratio=None,
                   cbar=True, cbar_shrink=1.0, cbar_title='',
                   xlabel=None, ylabel=None, x_rotation=0,
                   vmin=-3, vmax=3, figsize=None, fontsize=2):
    
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


def generate_new_model(adata=None, batch_key=None, n_layers=1, max_epochs=400):
    
    """
    Generates scVI model and sets latent representation in adata.obsm["X_scVI"]
    """

    ## Dedicated scVI model from data
    scvi.model.SCVI.setup_anndata(adata, batch_key=batch_key)

    model = scvi.model.SCVI(adata, n_layers=n_layers)
    model.train(max_epochs=max_epochs)

    latent = model.get_latent_representation()
    adata.obsm["X_scVI"] = latent
    
    return adata


def plot_cats(adata=None, groupby=None, basis='umap', sep_cats=True, dotsize=4, legend_loc='on data', figsize=None, fontsize=8):
    """
    Plots categorical annotations from obs onto embedding. 
    By default, it will make one subplot for each category, but set sep_cats to False to put them all on the same plot.
    Returns fig, axs
    """
    
    if sep_cats:

        samples = sorted(adata.obs[groupby].unique().tolist())
        numSamples = len(samples)
        ncols = 4
        nrows = int(np.ceil(numSamples/ncols))
        
        if figsize is None:
            figsize = (ncols, nrows)

        fig, axs = plt.subplots(ncols= ncols,
                                nrows= nrows,
                                sharex= True,
                                sharey= True,
                                gridspec_kw= {'hspace':0.1, 'wspace':0.1},
                                figsize= figsize)

        for s, ax in zip(samples, axs.flat):
            
            _= sc.pl.embedding(adata,
                               basis= basis,
                               color= groupby,
                               groups= s,
                               legend_loc= None,
                               frameon= False,
                               title= '',
                               show= False,
                               ax= ax)
            
            plot_dict = {'title': s, 'fontsize': fontsize, 'pad': 1}
            _= fix_plot(ax, plot_dict=plot_dict)

        return fig, axs

    else:

        if figsize is None:
            figsize = (2,2)

        fig,axs = plt.subplots(figsize= figsize)

        sc.pl.embedding(adata,
                        basis= basis,
                        color= groupby,
                        frameon= False,
                        add_outline= True,
                        size= dotsize,
                        legend_loc= legend_loc,
                        legend_fontsize= fontsize,
                        legend_fontweight= 'regular',
                        legend_fontoutline= 0,
                        show= False,
                        ax= axs)

        return fig, axs
