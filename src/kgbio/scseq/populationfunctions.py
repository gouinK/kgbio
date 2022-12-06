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
import sklearn.metrics
from statannot import add_stat_annotation

from matplotlib import pyplot as plt
from matplotlib.legend import Legend
import matplotlib.gridspec as gridspec

from kgbio.scseq import generalfunctions as gf

import warnings
warnings.filterwarnings("ignore")


def pct_df(adata=None, groupby='leiden', rep=None, xcat=None, drop_na=True, thresh=None, normalization=None, receptor_column=None):
    
    if rep is None:

        tx = adata.obs[xcat].unique().tolist()
        grps = adata.obs[groupby].unique().tolist()

        df = pd.DataFrame(columns=[xcat, groupby, 'percent'])

        idx = pd.Index(tx, name=xcat)

    else:

        pt = adata.obs[rep].unique().tolist()
        tx = adata.obs[xcat].unique().tolist()
        grps = adata.obs[groupby].unique().tolist()

        df = pd.DataFrame(columns=[rep, xcat, groupby, 'percent'])

        idx = pd.MultiIndex.from_product([pt, tx], names=[rep, xcat])

    if thresh is None:
        cellthresh = len(grps)
    else:
        cellthresh = thresh


    for i in idx:

        if rep is None:
            total = adata[(adata.obs[idx.name]==i)].obs
        else:
            total = adata[(adata.obs[idx.names[0]]==i[0])&(adata.obs[idx.names[1]]==i[1])].obs

        if (normalization=='clonotype'):
            total.drop_duplicates(subset=receptor_column, inplace=True)

        totalsize = len(total)

        if (totalsize>=cellthresh):

            counts = total[groupby].value_counts()

            counts.index = counts.index.tolist()
            ms = [x for x in grps if x not in counts.index]
            for m in ms:
                counts.loc[m] = 0
            counts = counts.loc[grps]

            counts = (counts/totalsize)*100
  
            counts = pd.DataFrame(counts.to_numpy(), index=counts.index, columns=['percent'])
            if rep is None:
                counts[xcat] = i
            else:
                counts[rep] = i[0]
                counts[xcat] = i[1]
            counts[groupby] = counts.index.tolist()
            counts.reset_index(inplace=True)
            counts.drop(columns='index', inplace=True)

            df = df.append(counts)

    
    if drop_na:
        df.dropna(how='any',inplace=True)
    
    return df


def shannon(adata=None, groupby='leiden', rep=None, xcat=None, drop_na=True, thresh=None, normalization='cells', receptor_column=None):
    """
    calculate normalized shannon entropy (SE) clonality
    clonality = 1 - (SE/ln(# of tcrs))
    """

    pt = adata.obs[rep].unique().tolist()
    tx = adata.obs[xcat].unique().tolist()

    idx = pd.MultiIndex.from_product([pt ,tx], names=[rep, xcat])
    df = pd.DataFrame(index=idx, columns=['shannon'])

    st = adata.obs[groupby].unique().tolist()
    totalgrps = len(st)
    
    if thresh is None:
        thresh = totalgrps

    for i in df.index:
        
        total = adata[(adata.obs[idx.names[0]]==i[0])&(adata.obs[idx.names[1]]==i[1])].obs

        if (normalization=='clonotype'):
            total.drop_duplicates(subset=receptor_column, inplace=True)

        totalsize = len(total)
        
        if (totalsize>=thresh):

            grps = total[groupby].value_counts()
            
            grps = grps/totalsize
            
            runningSum = 0
            
            for g in grps:
                runningSum += -1*(g * np.log(g))

            df.loc[i,'shannon'] = (1 - (runningSum/np.log(totalgrps)))

    df.reset_index(inplace=True)

    if drop_na:
        df.dropna(how='any', inplace=True)

    return df
    

def pct_comparison(adata=None,df=None,groupby='leiden',rep=None,xcat=None,hcat=None,ycat=None,normalization='cells',receptor_column=None,logic=None,
                    xorder=None,horder=None,show_stats=True,calc_pct=False,calc_shannon=False,drop_na=True,thresh=None,
                    dotsize=2,fontsize=4,ylim=None,figsize=None,return_df=False,size_max=None,vmin=-1,vmax=1,plot_type='boxplot',tight=False):
                
    tmp = adata.copy()

    if isinstance(xcat,dict):

        tmp.obs['marker'] = pd.DataFrame(index=tmp.obs.index, columns=['marker'])
        
        for element in xcat:

            comparator = xcat[element]

            if (logic=='and'):
                tmp.obs.loc[[all(tmp.obs.loc[x,y]==comparator[y] for y in comparator) for x in tmp.obs.index],'marker'] = element

            elif (logic=='or'):
                tmp.obs.loc[[any(tmp.obs.loc[x,y]==comparator[y] for y in comparator) for x in tmp.obs.index],'marker'] = element

        xcat = 'marker'

    if calc_pct:
        df = pct_df(tmp, groupby=groupby, rep=rep, xcat=xcat, drop_na=drop_na, thresh=thresh, normalization=normalization)
        ycat = 'percent'
        print('calculated percent')
    elif calc_shannon:
        df = shannon(tmp, groupby=groupby, rep=rep, xcat=xcat, drop_na=drop_na, thresh=thresh, normalization=normalization, receptor_column=receptor_column)
        ycat = 'shannon'
    else:
        if drop_na:
            df.dropna(how='any',inplace=True)

    if xorder is None:
        xorder = df[xcat].unique().tolist()
        cols = xorder
    if horder is None and hcat is not None:
        horder = df[hcat].unique().tolist()

    if (plot_type=='boxplot'):

        if (groupby is None) | (calc_shannon):
        
            combs = list(itertools.combinations(horder,2))

            if figsize is None:
                figsize = (2,2)

            fig,axs = plt.subplots(figsize=figsize)

            _= sns.boxplot(x=xcat,
                           y=ycat,
                           hue=hcat,
                           data=df,
                           color='w',
                           linewidth=0.5,
                           fliersize=0,
                           palette='colorblind',
                           order=xorder,
                           hue_order=horder,
                           boxprops=dict(alpha=0.2),
                           ax=axs)
            
            if show_stats:
                try:
                    test_results = add_stat_annotation(axs,
                                                       x=xcat,
                                                       y=ycat,
                                                       hue=hcat,
                                                       data=df,
                                                       order=xorder,
                                                       hue_order=horder,
                                                       box_pairs = [tuple((x,y) for y in c) for c in combs for x in xorder],
                                                       test='Mann-Whitney',
                                                       comparisons_correction=None,
                                                       text_format='simple',
                                                       loc='outside',
                                                       verbose=0,
                                                       fontsize=4,
                                                       linewidth=0.5,
                                                       line_height=0.01,
                                                       text_offset=0.01)
                except:
                    pass

            _= sns.stripplot(x=xcat,
                             y=ycat,
                             hue=hcat,
                             dodge=True,
                             jitter=0.1,
                             data=df,
                             size=dotsize,
                             order=xorder,
                             hue_order=horder,
                             ax=axs)

            if ylim is None:
                ymin = df[ycat].min() - 0.1*df[ycat].min()
                ymax = df[ycat].max() + 0.1*df[ycat].max()
            
            plot_dict = {
                         'ylim': (ymin, ymax),
                         'xlabel': '',
                         'ylabel': ycat,
                         'fixlegend': True,
                         'xticks': np.arange(len(xorder)),
                         'xticklabels': xorder,
                         'x_rotation': 90,
                         'fontsize': fontsize,
                         }

            _= gf.fix_plot(ax=axs, plot_dict=plot_dict)
            
            if tight:
                plt.tight_layout()

        else:
        
            st = df[groupby].unique().tolist()
            
            combs = list(itertools.combinations(horder, 2))
            
            if figsize is None:
                figsize = (len(st),1)
   
            fig,axs = plt.subplots(nrows=1,
                                   ncols=len(st),
                                   sharex=False,
                                   sharey=False,
                                   figsize=figsize)

            for s,ax in zip(st, axs.flat):

                tmp = df.loc[[x for x in df.index if df.loc[x,groupby]==s],:]

                if (tmp[ycat].sum()==0):
                    continue

                _= sns.boxplot(x=xcat,
                               y=ycat,
                               hue=hcat,
                               data=tmp,
                               color='w',
                               linewidth=0.5,
                               fliersize=0,
                               palette='colorblind',
                               order=xorder,
                               hue_order=horder,
                               boxprops=dict(alpha=0.2),
                               ax=ax)
                
                if show_stats:
                    try:
                        test_results = add_stat_annotation(ax,
                                                           x=xcat,
                                                           y=ycat,
                                                           hue=hcat,
                                                           data=tmp,
                                                           order=xorder,
                                                           hue_order=horder,
                                                           box_pairs = [tuple((x,y) for y in c) for c in combs for x in xorder],
                                                           test='Mann-Whitney',
                                                           comparisons_correction=None,
                                                           text_format='simple',
                                                           loc='outside',
                                                           verbose=0,
                                                           fontsize=4,
                                                           linewidth=0.5,
                                                           line_height=0.01,
                                                           text_offset=0.01)
                    except:
                        pass

                _= sns.stripplot(x=xcat,
                                 y=ycat,
                                 hue=hcat,
                                 dodge=True,
                                 jitter=0.1,
                                 data=tmp,
                                 size=dotsize,
                                 order=xorder,
                                 hue_order=horder,
                                 ax=ax)
                
                if ylim is None:
                    ymin = df[ycat].min() - 0.1*df[ycat].min()
                    ymax = df[ycat].max() + 0.1*df[ycat].max()
                
                plot_dict = {
                            'ylim': (ymin, ymax),
                            'xlabel': '',
                            'ylabel': ycat,
                            'fixlegend': True,
                            'xticks': np.arange(len(xorder)),
                            'xticklabels': xorder,
                            'x_rotation': 90,
                            'fontsize': fontsize,
                            }

                _= gf.fix_plot(ax=axs, plot_dict=plot_dict)
            
            if tight:
                plt.tight_layout()
            
    elif (plot_type=='heatmap'):

        st = df[groupby].unique().tolist()

        effect_size = pd.DataFrame(index=st, columns=cols, dtype=np.float64)
        pval = pd.DataFrame(index=st, columns=cols, dtype=np.float64)
        
        for s in st:
            for c in cols:
                
                y0 = df[(df[groupby]==s)&(df[xcat]==c)&(df[hcat]==horder[0])][ycat].to_numpy(dtype=np.float64)
                y1 = df[(df[groupby]==s)&(df[xcat]==c)&(df[hcat]==horder[1])][ycat].to_numpy(dtype=np.float64)
                S,P = scipy.stats.ranksums(y0,y1)
                
                effect_size.loc[s,c] = S
                pval.loc[s,c] = P
        
        pval = -1*np.log10(pval.to_numpy(dtype=np.float64))
        pval[np.isinf(pval)] = np.max(pval[~np.isinf(pval)])
        pval = pd.DataFrame(pval,index=st,columns=cols)
        
        d = scipy.spatial.distance.pdist(effect_size.to_numpy(dtype='float64'), metric='euclidean')
        l = scipy.cluster.hierarchy.linkage(d, metric='euclidean', method='complete', optimal_ordering=True)
        dn = scipy.cluster.hierarchy.dendrogram(l, no_plot=True)
        order = dn['leaves']
        
        effect_size = effect_size.iloc[order,:]
        pval = pval.iloc[order,:]
        
        if size_max is None:
            size_max = pval.max().item()

        ref_s = [1.30,size_max]
        ref_l = ['0.05','maxsize: '+'{:.1e}'.format(10**(-1*size_max))]

        if figsize is None:
            figsize = (2, len(st))
        
        fig,axs = plt.subplots(nrows=1,
                               ncols=2,
                               figsize=figsize)
        
        gf.heatmap2(effect_size,
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
        _= gf.fix_plot(axs[1], plot_dict=plot_dict)
        
    elif (plot_type=='pie_chart'):

        grps = df[groupby].unique().tolist()

        xcats = df[xcat].unique().tolist()

        if figsize is None:
            figsize = (len(xcats), 1)

        fig,axs = plt.subplots(nrows=1,
                               ncols=len(xcats),
                               sharex=True,
                               sharey=True,
                               figsize=figsize)

        for x,ax in zip(xcats, axs.flat):

            tmp = df.query(" {0}==@x ".format(xcat))
            tmp.set_index(groupby, inplace=True)
            tmp = tmp.loc[grps,'percent'].to_numpy()

            _= ax.pie(tmp,
                      labels=grps,
                      rotatelabels=True,
                      textprops={'fontsize':fontsize},
                      wedgeprops={'linewidth':0})
            
            plot_dict = {'title': x, 'fontsize': fontsize}
            _= gf.fix_plot(ax, plot_dict=plot_dict)

    if return_df:
        return fig, axs, df
    else:
        return fig, axs



## pie chart of cluster distribution for each category
def custom_pie(adata=None, comparator=None, reference=None, groupby='leiden', labels=None, fontsize=None, figsize=None):

    samp1 = adata[[all(adata.obs.loc[x,y]==comparator[y] for y in comparator) for x in adata.obs.index]]
    samp2 = adata[[all(adata.obs.loc[x,y]==reference[y] for y in reference) for x in adata.obs.index]] 

    samples = [samp1, samp2]

    if labels is None:
        labels = ['comparator', 'reference']
    
    grps = sorted(adata.obs[groupby].unique().tolist())
    
    if fontsize is None:
        fontsize = 4
    if figsize is None:
        figsize = (4,2)

    fig,axs = plt.subplots(nrows=1,
                           ncols=2,
                           sharex=True,
                           sharey=True,
                           figsize=figsize)

    for label,tmp,ax in zip(labels, samples, axs.flat):

        counts = tmp.obs[groupby].value_counts()
        counts.index = counts.index.tolist()

        ms = [x for x in grps if x not in counts.index] 
        for m in ms:
            counts.loc[m] = 0

        counts = counts.loc[grps]

        ax.pie(counts,
               labels=grps,
               normalize=True,
               rotatelabels=True,
               textprops={'fontsize':fontsize},
               wedgeprops={'linewidth':0})

        plot_dict = {'title': label, 'fontsize': fontsize}
        _= gf.fix_plot(ax, plot_dict=plot_dict)
        
    return fig,axs


def cmp_classifications(adata=None, sampleid=None, reference=None, comparators=None, classifications=None, metrics=None, pattern=None):

    """
    By default, calculates mutual information, rand index, and % match for labels between two datasets.
    Cell barcode pattern (adata.obs index) can be specified, otherwise defaults to '[ACGT]{16,18}'.

    """

    if metrics is None:
        metrics = ['AMI', 'ARI', 'pct_match']
    if pattern is None:
        pattern='[ACGT]{16,18}'

    if not isinstance(comparators, list):
        comparators = [comparators]
    if not isinstance(metrics, list):
        metrics = [metrics]
    if not isinstance(classifications, list):
        classifications = [classifications]
    
    cols = pd.MultiIndex.from_product([metrics, classifications], names=['metrics', 'classifications']) 

    df = pd.DataFrame(index=comparators, columns=cols, dtype=np.float64)

    for comp in comparators:

        ref_clust = adata.obs.query(f" {sampleid} == @reference ")
        ref_clust.index = [re.search(pattern=pattern, string=x).group() for x in ref_clust.index]

        comp_clust = adata.obs.query(f" {sampleid} == @comp ")
        comp_clust.index = [re.search(pattern=pattern, string=x).group() for x in comp_clust.index]

        BC = [x for x in ref_clust.index if x in comp_clust.index]

        ref_clust = ref_clust.loc[BC,:]
        comp_clust = comp_clust.loc[BC,:]

        for col in cols:

            m = col[0]
            c = col[1]

            ref_tmp = ref_clust.loc[:,c]
            comp_tmp = comp_clust.loc[:,c]

            if m == 'AMI':
                val = sklearn.metrics.adjusted_mutual_info_score(ref_tmp, comp_tmp)
            elif m == 'ARI':
                val = sklearn.metrics.adjusted_rand_score(ref_tmp, comp_tmp)
            elif m == 'pct_match':
                val = np.count_nonzero(ref_tmp == comp_tmp) / len(ref_tmp)
            else:
                print(f'no metric definition for {m}')
                break

            df.loc[comp,col] = val

    return df