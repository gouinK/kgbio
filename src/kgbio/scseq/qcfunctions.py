import os
import re
import math
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

from matplotlib import pyplot as plt
from matplotlib.legend import Legend
import matplotlib.gridspec as gridspec

from kgbio.utils.plotting import fix_plot

def collect_cellr_metrics(inpath=None):
	"""
	Constructs dataframe of cellranger metrics from a directory containing
	one or more cellranger output folders.

	The expected file structure is: {inpath}/{sample_x}/outs/metrics_summary.csv

	Returns dataframe
	"""
	
	if inpath is None:
		print('error no path')
		return

	files = os.listdir(inpath)
	files = sorted([x for x in files if '.DS' not in x])

	dfList = []
	for f in files:

		metrics = pd.read_csv(f'{inpath}/{f}/outs/metrics_summary.csv', index_col=None, header=0)
		metrics.index = [f]
		dfList.append(metrics)

	df = pd.concat(dfList)

	df.columns = [x.replace(' ','_') for x in df.columns]

	for row in df.index:
		for col in df.columns:
			if isinstance(df.loc[row,col],str):
				df.loc[row,col] = float(df.loc[row,col].replace('%','').replace(',',''))
			else:
				df.loc[row,col] = float(df.loc[row,col])

	return df


def plot_cellR_metrics(df=None, sampleid=None, metrics=None, hues=None, dotsize=4, fontsize=4, markerscale=0.1):
	"""
	Makes scatterplots of cellranger metrics contained in a dataframe.

	Returns fig, axs
	"""

	if metrics is None:
		metrics = ['Estimated_Number_of_Cells', 'Sequencing_Saturation', 'Median_Genes_per_Cell', 'Total_Genes_Detected', 'Median_UMI_Counts_per_Cell']
	if hues is None:
		hues = ['library_source', 'Q30_Bases_in_Barcode', 'Q30_Bases_in_UMI', 'Q30_Bases_in_RNA_Read']
	if sampleid is None:
		print('must provide name of sample column')
		return

	forder = [(x,y) for x in metrics for y in hues]

	fig,axs = plt.subplots(nrows=len(metrics),
						   ncols=len(hues),
						   sharex=True,
						   sharey=False,
						   figsize=(2*len(hues), 2*len(metrics)))

	numSamples = len(df[sampleid].unique().tolist())

	for f,ax in zip(forder, axs.flat):
	    
		m = f[0]
		h = f[1]

		if h==sampleid:
			if numSamples <= 10:
				palette = 'tab10'
			else:
				palette = 'tab20'
		else:
			palette = 'Reds'
			
		_= sns.scatterplot(x='Number_of_Reads',
						   y=m,
						   hue=h,
						   data=df,
						   s=dotsize,
						   palette=palette,
						   ax=ax)

		plot_dict = {'ylabel': '', 'yticks': [], 'legend': False, 'fontsize': fontsize}
		_= fix_plot(ax, plot_dict=plot_dict)

		if ax.is_first_row():
			_= ax.set_title(h, fontsize=fontsize)
		if ax.is_last_row():
			_= ax.set_xlabel('Number_of_Reads', fontsize=fontsize)
			_= ax.legend().set_visible(True)
			_= ax.legend(markerscale=markerscale, fontsize=fontsize/2, frameon=False)
		if ax.is_first_col():
			_= ax.set_ylabel(m, fontsize=fontsize)

	return fig,axs


def calc_sparcity(adata=None):
	"""
	Calculates sparcity of each gene (fraction of zeros).

	Returns array
	"""
	
	nnz = adata.X.count_nonzero()
	total = adata.X.shape[0] * adata.X.shape[1]
	sparcity = (1-(nnz/total))*100

	return sparcity


def annotate_genes(adata=None, remove_noncoding=False, species=None):
	"""
	Annotates genes using biomart.
	Included annotations are: 
		attrs = [
				"ensembl_gene_id", "hgnc_symbol", 
				"start_position", "end_position", 
				"chromosome_name", "percentage_gene_gc_content",
				"gene_biotype"
			]
	Can optionally remove non-coding genes from anndata.
	
	Species should be: “hsapiens”, “mmusculus”, “drerio”, etc

	Returns anndata with updated var
	"""

	print(f'original adata shape: {adata.shape}')

	if "ENS" in adata.var_names[0]:
		gene_name_type = "ensembl_gene_id"
	else:
		gene_name_type = "hgnc_symbol"

	attrs = [
				"ensembl_gene_id", "hgnc_symbol", 
				"start_position", "end_position", 
				"chromosome_name", "percentage_gene_gc_content",
				"gene_biotype"
			]

	annot = sc.queries.biomart_annotations(org=species, attrs=attrs)
	annot.dropna(how='any', subset=[gene_name_type], inplace=True)
	annot.drop_duplicates(subset=[gene_name_type], keep='first', inplace=True)
	annot = annot.loc[~annot['chromosome_name'].str.contains("CHR"),:]
	annot['gene_length'] = annot['end_position'] - annot['start_position']
	annot.set_index(gene_name_type,inplace=True)

	adata.var = adata.var.merge(annot, how='left', left_index=True, right_index=True)

	keepTypes = [
					'protein_coding',
					'IG_V_gene', 'IG_D_gene', 'IG_C_gene', 'IG_J_gene',
					'TR_V_gene', 'TR_J_gene', 'TR_C_gene' 'TR_D_gene'
				]

	if remove_noncoding:
		adata = adata[:, adata.var.query(" gene_biotype in @keepTypes ").index].copy()

	print(f'final adata shape: {adata.shape}')

	return adata


def qc_filtering(adata=None, percentile=5, mtThresh=10, scrublet=True, do_plot=False, strip_plot=False, do_filter=True):
	"""
	Performs filtering of cells based on percentile of total counts, number of unique genes, and mitochondrial percentage.
	Also optionally performs scrublet filtering.
	Also optionally provides violin plots of 'n_genes_by_counts', 'total_counts', 'pct_counts_mt' on dataset prior to filtering.
	
	Returns anndata with filtered cells removed.
	"""
	
	if scrublet:
		doublet_rate = 0.07
		sc.external.pp.scrublet(adata, expected_doublet_rate=doublet_rate)
		if np.sum(pd.isnull(adata.obs['predicted_doublet'])) > 0:
			print("Found NaN result from scrublet, not filtering out doublets, please check manually.")
			adata.obs.drop(columns=['doublet_score','predicted_doublet'], inplace=True)
		else:
			print(f"Found {np.sum(adata.obs['predicted_doublet'])} doublets, now removing.")
			adata = adata[~adata.obs['predicted_doublet']].copy()

	geneThresh = np.percentile(adata.obs.n_genes_by_counts, q=percentile)
	umiThresh = np.percentile(adata.obs.total_counts, q=percentile)

	print(f"Unique gene count threshold = {geneThresh:.0f}")
	print(f"UMI count threshold = {umiThresh:.0f}")
	print(f"Mitochondrial percent threshold = {mtThresh:.0f}%")

	if do_plot:

		metrics = ['n_genes_by_counts', 'total_counts', 'pct_counts_mt']
		ylims = [5000, 10000, 100]
		steps = [200, 500, 10]

		fig,axs = plt.subplots(nrows= 3,
							   ncols= 1,
							   sharex= True,
							   sharey= False,
							   figsize= (4,3))

		for metric, ylim, step, ax in zip(metrics, ylims, steps, axs.flat):

			_= sns.violinplot(y= metric,
							  x= 'sampleID',
							  data= adata.obs,
							  inner= 'quartile',
							  ax= ax)
			
			if strip_plot:
				_= sns.stripplot(y= metric,
								 x= 'sampleID',
								 data= adata.obs,
								 size= 0.1,
								 color= 'k',
								 ax= ax)
			
			plot_dict = {
							'ylim': (0, ylim),
							'yticks': np.arange(start=0, stop=ylim, step=step),
							'xticklabels': adata.obs.sampleID.unique().tolist(),
							'x_rotation': 90,
							'fontsize': 4
						}
							
			_= fix_plot(ax, plot_dict=plot_dict)

		plt.tight_layout()

	if do_filter:

		print(f'pre filter {len(adata)}')
		adata = adata[adata.obs.query(" n_genes_by_counts >= @geneThresh & total_counts >= @umiThresh & pct_counts_mt < @mtThresh ").index].copy()
		print(f'post filter {len(adata)}')

	return adata


def spatial_plots(adata=None, outdir=None):
	"""
	Generates QC plots ('total_counts', 'n_genes_by_counts', 'pct_counts_mt') on Visium image.
	"""
    
	sc.pl.spatial(adata,
                  img_key= 'hires',
                  color= ['total_counts', 'n_genes_by_counts', 'pct_counts_mt'],
                  vmin= 0,
                  vmax= 'p99',
                  ncols= 3,
                  cmap= 'inferno')
    
	plt.savefig(outdir+'/spatial_QC_metrics.png', dpi=600, bbox_inches='tight')

	sc.pl.spatial(adata, img_key='hires', color=['leiden'])

	plt.savefig(outdir+'/spatial_leiden.png', dpi=600, bbox_inches='tight')

	return


def run_celltypist(adata=None, model=None, majority_voting=None, mode=None, p_thres=0.5):
	"""
	Performs celltypist automated cell type identification.
	By default, will run majority voting with prob match mode.
	Returns adata with cell type labels in obs.
	"""
	
	if (majority_voting is None) | (mode is None):
		majority_voting = True
		mode = 'prob match'

	celltypist.models.download_models(force_update = True)

	model_info = celltypist.models.Model.load(model = model)
	print(f"Using {model}:")
	print(model_info)
	print("\nAvailable cell type labels:")
	print(model_info.cell_types)

	predictions = celltypist.annotate(filename= adata,
									  model= model,
									  majority_voting= majority_voting,
									  mode= mode,
									  p_thres= p_thres)

	adata2 = predictions.to_adata()

	return adata2