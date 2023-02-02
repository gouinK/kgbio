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
from smart_open import open as sopen
from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib.legend import Legend
import matplotlib.gridspec as gridspec


def stream_fastq(fqfile):
    """Read a fastq file and provide an iterable of the sequence ID, the
    full header, the sequence, and the quaity scores.
    Note that the sequence ID is the header up until the first space,
    while the header is the whole header.
    """
    qin = sopen(fqfile, 'rt')

    while True:
        header = qin.readline()
        if not header:
            break
        header = header.strip()
        seqidparts = header.split(' ')
        seqid = seqidparts[0]
        seq = qin.readline()
        seq = seq.strip()
        _ = qin.readline()
        qualscores = qin.readline()
        qualscores = qualscores.strip()
        header = header.replace('@', '', 1)
        yield seqid, header, seq, qualscores


def readfastq(inpath):
    """
    Return SeqIO record dictionary (supports gzip and non-gzip fastq.)
    """
    if '.gz' in inpath:
        with gzip.open(inpath, "rt") as handle:
            record_dict = SeqIO.to_dict(SeqIO.parse(handle,'fastq'))            
    else:
        record_dict = SeqIO.to_dict(SeqIO.parse(inpath,'fastq'))
    
    return record_dict


def generate_read_df(inpath=None, attrs=None, cycle_idx=None, suffix=None):
    
    '''
    Generates pandas dataframe from fastq with optional attributes included.
    
    By default only provides run, flowcell, lane, tile, and coordinates.
    
    Optional attributes to include are: cycleQS, seq, QS
    
    If suffix is provided, the original readID will be stored as a new column
    and a new index will be generated with the following format <readID>_<suffix>.
    '''
    
    print(f"reading {inpath}")
    
    if attrs is None:
        attrs = []

    record_dict = readfastq(inpath)
    tmp = list(record_dict.keys())[0].split(':')

    if len(tmp) == 7:
        
        cols = ['test','run','flowcell','lane','tile','x_coord','y_coord']
        tmp = pd.DataFrame([x.split(':') for x in record_dict], index=list(record_dict.keys()), columns=cols)
    
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
    
    if suffix is not None:
        tmp['readID'] = tmp.index.tolist()
        tmp.rename(index='{}_{}'.format(suffix))

    return tmp