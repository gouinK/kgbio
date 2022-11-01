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
## TODO: fix gseapy dependency
# import gseapy as gp
import seaborn as sns
from glob import glob
import sklearn.metrics
from statannot import add_stat_annotation

from matplotlib import pyplot as plt
from matplotlib.legend import Legend
import matplotlib.gridspec as gridspec

from kgbio.scseq import generalfunctions as gf


def leiden_dgex(adata=None, groupby='leiden', use_raw=False, method='wilcoxon', pts=True, tie_correct=True):

    sc.tl.rank_genes_groups(adata,
                            groupby=groupby,
                            use_raw=use_raw,
                            method=method,
                            pts=pts,
                            tie_correct=tie_correct)
    
    grps = adata.obs[groupby].unique().tolist()

    dfList = []
    for g in grps:
        tmp = sc.get.rank_genes_groups_df(adata, group=str(g))
        tmp['group'] = g
        dfList.append(tmp)

    dgex = pd.concat(dfList)

    return dgex


def cepo_dgex(adata=None, groupby='leiden'):

    st = adata.obs[groupby].unique().tolist()
    nSubs = len(st)
    sdf = pd.DataFrame(index=adata.var_names, columns=st)

    for s in st:
        tmp = adata[(adata.obs[groupby]==s)].copy()
        
        nCells = tmp.shape[0]
        nGenes = tmp.shape[1]
        
        nz = tmp.X.getnnz(axis=0)/nCells
        cv = np.std(tmp.X.toarray(),axis=0)/tmp.X.mean(axis=0)
        
        x1 = scipy.stats.rankdata(nz)/(nGenes+1)
        x2 = 1 - (scipy.stats.rankdata(cv)/(nGenes+1))
        
        sdf.loc[:,s] = np.mean([x1,x2],axis=0)

    finaldf = pd.DataFrame(index=adata.var_names,columns=st)

    for s in st:
        runningSum = np.zeros(nGenes)
        for i in [x for x in st if x!=s]:
            runningSum += (sdf.loc[:,s] - sdf.loc[:,i]).to_numpy()
        finaldf.loc[:,s] = runningSum/(nSubs-1)
        
    return finaldf


def custom_dgex(adata=None, comparator=None, reference=None, grouping_type='custom', groupby=None, logic='and', bycluster=True, filter_pval=True):

    """
    if grouping_type is 'normal', then creates annotations based on groups from one obs column (groupby).


    if grouping_type is 'custom', then creates custom annotations based on arbritrary groupings of obs columns.
    the comparator and reference should be dictionaries with obs columns as keys and desired groups as values.
    the logic can be set to "and" which means all conditions in the dictionary must be true,
    or it can be set to "or" which means that only one of the conditions in the dictionary must be true.
    
    if reference is set to None, then the reference will be all cells not contained in the comparator.

    """

    tmp = adata.copy()

    tmp.obs['dgex_marker'] = pd.DataFrame(index=tmp.obs.index, columns=['dgex_marker'])

    if (grouping_type=='normal'):
        
        tmp.obs.loc[[x for x in tmp.obs.index if tmp.obs.loc[x,groupby] in comparator],'dgex_marker'] = 'comparator'
        tmp.obs.loc[[x for x in tmp.obs.index if tmp.obs.loc[x,groupby] in reference],'dgex_marker'] = 'reference'
    
    elif (grouping_type=='custom'):
        
        if (logic=='and'):
            tmp.obs.loc[[all(tmp.obs.loc[x,y]==comparator[y] for y in comparator) for x in tmp.obs.index],'dgex_marker'] = 'comparator'
            if reference is None:
                tmp.obs.loc[(tmp.obs.marker!='comparator'),'marker'] = 'reference'
            else:
                tmp.obs.loc[[all(tmp.obs.loc[x,y]==reference[y] for y in reference) for x in tmp.obs.index],'dgex_marker'] = 'reference'
        elif (logic=='or'):
            tmp.obs.loc[[any(tmp.obs.loc[x,y]==comparator[y] for y in comparator) for x in tmp.obs.index],'dgex_marker'] = 'comparator'
            if reference is None:
                tmp.obs.loc[(tmp.obs.marker!='comparator'),'marker'] = 'reference'
            else:
                tmp.obs.loc[[any(tmp.obs.loc[x,y]==reference[y] for y in reference) for x in tmp.obs.index],'dgex_marker'] = 'reference'
    
    print(tmp.obs.dgex_marker.value_counts())

    sc.tl.rank_genes_groups(tmp,
                            groupby='dgex_marker',
                            use_raw=False,
                            groups=['comparator'],
                            reference='reference',
                            method='wilcoxon',
                            pts=True,
                            tie_correct=True)

    dgex = sc.get.rank_genes_groups_df(tmp, group='comparator')
    dgex['group'] = 'all'

    if bycluster:

        grps = tmp.obs.leiden.unique().tolist()

        for g in grps:
            
            tmp2 = tmp[(tmp.obs.leiden==g)].copy()
            compSize = len(tmp2[(tmp2.obs.dgex_marker=='comparator')])
            refSize = len(tmp2[(tmp2.obs.dgex_marker=='reference')])
            
            if (compSize>1)&(refSize>1):
                
                sc.tl.rank_genes_groups(tmp2,
                                        groupby='dgex_marker',
                                        use_raw=False,
                                        groups=['comparator'],
                                        reference='reference',
                                        method='wilcoxon',
                                        pts=True,
                                        tie_correct=True)
                
                df = sc.get.rank_genes_groups_df(tmp2, group='comparator')
                df['group'] = g
                
                dgex = dgex.append(df)

    if filter:
        dgex = dgex.query(" pvals <= 0.05 ").copy()

    return dgex


def dgex_plot(adata=None, dgex=None, groupby='leiden', topn=20, pvalCutoff=0.05, fcCutoff=1, pctCutoff=0.3, use_FDR=True,
              dendro=False, plot_type='dotplot', cmap='Reds', figsize=None, vmin=0, vmax=1, fontsize=4, size_max=None):
    
    """
    plot type options are: dotplot, matrixplot, scanpy_heatmap, custom_heatmap
    
    custom_heatmap returns effect_size,pval,fig,axs
    all others return fig
    
    """

    grp = sorted(adata.obs[groupby].unique().tolist())
    
    GOI = []
    
    for g in grp:
        
        if use_FDR:
            GOI += dgex[(dgex.pvals_adj<=pvalCutoff)&
                        (dgex.logfoldchanges>=fcCutoff)&
                        (dgex.pct_nz_group>=pctCutoff)&
                        (dgex.group==int(g))].sort_values(by='scores',ascending=False).names[:topn].tolist()
        else:
            GOI += dgex[(dgex.pvals<=pvalCutoff)&
                        (dgex.logfoldchanges>=fcCutoff)&
                        (dgex.pct_nz_group>=pctCutoff)&
                        (dgex.group==int(g))].sort_values(by='scores',ascending=False).names[:topn].tolist()

    _,idx = np.unique(GOI,return_index=True)

    GOI = [GOI[index] for index in sorted(idx)]
    
    tmp = adata[:,GOI].copy()
    
    if figsize is None:
        figsize = (2,2)

    if (plot_type=='custom_heatmap'):

        effect_size, pval, fig, axs = gf.plot_gex(tmp,
                                                  GOI=GOI,
                                                  dgex=dgex,
                                                  groupby=groupby,
                                                  dendro=dendro,
                                                  plot_type=plot_type,
                                                  cmap=cmap,
                                                  figsize=figsize,
                                                  vmin=vmin,
                                                  vmax=vmax,
                                                  fontsize=fontsize,
                                                  size_max=size_max)
        
        return effect_size, pval, fig, axs

    else:

        test = gf.plot_gex(tmp,
                           GOI=GOI,
                           groupby=groupby,
                           dendro=dendro,
                           plot_type=plot_type,
                           cmap=cmap,
                           figsize=figsize,
                           vmin=vmin,
                           vmax=vmax,
                           fontsize=fontsize)
        
        return test
    


# def pathway_enrich(rnk, genesets=None, test='enrich', org='Human', background=None, no_plot=True, outdir=None, ascending=False):

#     """
#     test can be either 'enrich' or 'gsea'.
#     'enrich' expects just a list of genes, and provide the appropriate background gene list.
#     'gsea' expects a dataframe with genes on the index and one column containing a ranking metric (specify ascending order).
#     if you want to run a mouse dataset, you need to convert the gene names to human.

#     """

#     if (test=='enrich'):

#         enr_results = gp.enrichr(gene_list=rnk,
#                                  gene_sets=genesets,
#                                  organism=org, 
#                                  description='test_name',
#                                  background=background,
#                                  outdir=outdir,
#                                  no_plot=no_plot,
#                                  cutoff=1)

#         enr_results = enr_results.results

#     elif (test=='gsea'):

#         enr_results = pd.DataFrame(columns=['es', 'nes', 'pval', 'fdr', 'geneset_size', 'matched_size', 'genes', 'ledge_genes', 'group'])

#         for g in genesets:
#             pre_res = gp.prerank(rnk=rnk,
#                                  gene_sets=g, 
#                                  processes=4,
#                                  permutation_num=100,
#                                  ascending=ascending,
#                                  outdir=outdir, 
#                                  format='png', 
#                                  seed=6,
#                                  no_plot=no_plot,
#                                  min_size=0,
#                                  max_size=500,
#                                  verbose=True)

#             pre_res = pre_res.res2d
#             pre_res['group'] = g
#             enr_results = enr_results.append(pre_res)

#     return enr_results

def filter_pathways(enr_results=None, significance='fdr', pval_cutoff=0.05, direction=None, test='enrich', add_blacklist=None, term_overlap=0.2, gene_overlap=0.5, fontsize=2, figsize=None, ax=None):

    blacklist = ['Homo','sapiens','Immune','immune','Cell','Cellular','cell','cellular','Pathway','pathway','Response','response','to','of','in']
    
    if add_blacklist is not None:
        blacklist += add_blacklist

    if (test=='enrich'):

        enr_results['overlapsize'] = [len(x.split(';')) for x in enr_results.Genes]
        enr_results['setsize'] = [int(x.split('/')[1]) for x in enr_results.Overlap]
        enr_results['fraction'] = enr_results['overlapsize']/enr_results['setsize']
        enr_results.index = enr_results.Term

        gene_col = 'Genes'

        if significance=='fdr':
            significance = 'Adjusted P-value'
        else:
            significance = 'P-value'

        enr_results = enr_results[(enr_results[significance]<=pval_cutoff)]

    elif (test=='GSEA'):

        enr_results['fraction'] = enr_results['matched_size']/enr_results['geneset_size']

        gene_col = 'genes'

        if (direction=='up'):
            enr_results = enr_results[(enr_results['nes']>0)&(enr_results[significance]<=pval_cutoff)]
        elif (direction=='down'):
            enr_results = enr_results[(enr_results['nes']<0)&(enr_results[significance]<=pval_cutoff)]

    tmp = enr_results

    tmp.sort_values(by=significance,inplace=True)


    terms = []

    for i,t in enumerate(tmp.index):

        if (i==0):
            terms = terms + [t]

        else:

            test = t.split(' ')
            test = [x for x in test if x not in blacklist]
            test = [x for x in test if 'R-HSA' not in x and 'GO:' not in x and 'hsa' not in x]

            counter = 0

            for j in terms:
                reference = j.split(' ')
                match = [x for x in reference if x in test]
                if (len(match) > term_overlap*len(test)):
                    counter = 1

            if (counter==0):
                terms = terms + [t]

    tmp = tmp.loc[terms,:]

    tmp.sort_values(by='fraction', ascending=False, inplace=True)
    genes = []
    terms = []
    for i,t in enumerate(tmp.index):

        if (i==0):
            genes = genes + [tmp.loc[t,gene_col]]
            terms = terms + [t]
        else:

            test = tmp.loc[t, gene_col].split(';')

            counter = 0

            for j in genes:
                reference = j.split(';')
                match = [x for x in reference if x in test]
                if (len(match) > gene_overlap*len(test)):
                    counter = 1

            if (counter==0):
                genes = genes + [tmp.loc[t,gene_col]]
                terms = terms + [t]

    tmp = tmp.loc[terms,:]
    tmp.sort_values(by=significance, ascending=False, inplace=True)


    df = pd.DataFrame(-1*np.log10(tmp[significance].to_numpy(dtype=np.float64)), index=tmp.index.tolist(), columns=[significance])
    df.loc[np.isinf(df[significance]),significance] = np.max(df.loc[~np.isinf(df[significance]),significance])

    if ax is None:

        if figsize is None:
            figsize=(2,2)

        fig,ax = plt.subplots(figsize=figsize)

    ys = np.arange(len(df))
    ws = df.to_numpy(dtype=np.float64).flatten()

    _= ax.barh(y=ys,
               width=ws,
               height=0.5,
               tick_label=df.index.tolist())
    
    plot_dict = {'yticklabels': df.index.tolist(), 'xlabel': '-log10(pval)', 'fontsize': fontsize, 'pad': 2}
    _= gf.fix_plot(ax, plot_dict=plot_dict)

    return ax,tmp