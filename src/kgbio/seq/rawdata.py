import re
import os
import math
import time
import gzip
import h5py
import pysam
import scipy
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from Bio import SeqIO
from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib.legend import Legend
import matplotlib.gridspec as gridspec

def readfastq(inpath):
    if '.gz' in inpath:
        with gzip.open(inpath, "rt") as handle:
            record_dict = SeqIO.to_dict(SeqIO.parse(handle,'fastq'))            
    else:
        record_dict = SeqIO.to_dict(SeqIO.parse(inpath,'fastq'))
    
    return record_dict


def generate_read_df(inpath=None, fmt='new', attrs=[], cycle_idx=None):
    
    '''
    Generates pandas dataframe from fastq with optional attributes included.
    By default only provides run, lane, tile, and coordinates.
    Optional attributes to include are: cycleQS, seq, QS
    '''
    
    print('reading '+inpath, flush=True)
    
    record_dict = readfastq(inpath)
    tmp = list(record_dict.keys())[0].split(':')

    if len(tmp) == 7:
        
        cols = ['test','run','flowcell','lane','tile','x_coord','y_coord']
        tmp = pd.DataFrame([x.split(':') for x in record_dict],index=list(record_dict.keys()),columns=cols)
    
    else:
        
        print("Error: unknown format.")
        return None
        
    tmp['x_coord'] = tmp['x_coord'].to_numpy(dtype=int)
    tmp['y_coord'] = tmp['y_coord'].to_numpy(dtype=int)
    
    if 'cycleQS' in attrs:
        tmp['cycleQS'] = [[record_dict[x].letter_annotations["phred_quality"][y] for y in [z for z in cycle_idx if z<len(record_dict[x])]] for x in record_dict]
        tmp[['cycle'+str(x+1)+'_QS' for x in cycle_idx]] = pd.DataFrame(tmp['cycleQS'].tolist(),index=tmp.index)
    if 'seq' in attrs:
        tmp['sequence'] = [str(record_dict[x].seq) for x in record_dict]
    if 'QS' in attrs:
        tmp['QS'] = [record_dict[x].letter_annotations["phred_quality"] for x in record_dict]
        
    return tmp