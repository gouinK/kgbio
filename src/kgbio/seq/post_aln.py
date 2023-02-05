import re
import os
import ray
import math
import time
import gzip
import h5py
import pysam
import scipy
import logging
import numpy as np
import pandas as pd
import seaborn as sns
from Bio import SeqIO


def get_seq(sites=None, ref=None):
    """
    Extracts sequences for provided sites from ref fasta file.
    Sites must be provided as a list of strings or a Series 
    in the following format: contig:start:stop

    Returns Series if Series was provided, otherwise returns list.
    """
    SeqDict = SeqIO.to_dict(SeqIO.parse(ref, "fasta"))
    
    if isinstance(sites, pd.Series):
        seqs = sites.apply(lambda x: str(SeqDict[x.split(':')[0]][int(x.split(':')[1]) : int(x.split(':')[2])].seq.upper()))
    else:
        seqs = pd.Series(sites).apply(lambda x: str(SeqDict[x.split(':')[0]][int(x.split(':')[1]) : int(x.split(':')[2])].seq.upper())).tolist()
        
    return seqs


def bed2df(inpath=None, remove_slop=False, include_seq=False, include_seq_noslop=False, ref=None):
    """
    Reads 3-column (contig, start, end) bed file (must have no header) and returns dataframe.
    The index will be constructed as follows: {contig}:{start}:{end}
    
    remove_slop can be set to True, and the slop will try to be inferred from the file name,
    or can be set to an integer to specify the amount of slop.
    
    include_seq and include_seq_noslop can be set to True to also extract the reference
    sequence at each site (with/without slop) in the bed file. ref must be provided for this to work.

    Returns dataframe.
    """
    
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    logging.info(f"Processing {inpath}.")

    df = pd.read_csv(f'{inpath}', index_col=None, header=None, sep='\t')
    df.columns = ['contig','start','end']
    df.index = df['contig'] + ':' + df['start'].astype(str) + ':' + df['end'].astype(str)
    
    if include_seq:
        if ref is None:
            logging.error("Must provide ref if include_seq is True.")
            return None
        
        logging.info("Extracting sequences.")
        df['seq'] = get_seq(sites= df.index, ref= ref)
            
    if remove_slop:
        if isinstance(remove_slop, bool):
            try:
                slop = re.findall(pattern='slop[0-9]+', string=inpath.split('/')[-1])
                if len(slop) > 0:
                    slop = int(slop[0][-1])
                    logging.info(f"Slop of {slop}bp found.")
                    df['start_noslop'] = df['start'] + slop
                    df['end_noslop'] = df['end'] - slop
                    
                    if include_seq_noslop:
                        logging.info("Extracting sequences with no slop.")
                        df['seq_noslop'] = get_seq(sites= df['contig'] + ':' + df['start_noslop'].astype(str) + ':' + df['end_noslop'].astype(str), ref= ref)
                        
            except:
                logging.info("No slop found.")
                pass
            
        else:
            slop = remove_slop
            logging.info(f"Slop of {slop}bp found.")
            df['start_noslop'] = df['start'] + slop
            df['end_noslop'] = df['end'] - slop
            
            if include_seq_noslop:
                logging.info(f"Extracting sequences with {slop}bp slop removed.")
                df['seq_noslop'] = get_seq(sites= df['contig'] + ':' + df['start_noslop'].astype(str) + ':' + df['end_noslop'].astype(str), ref= ref)
                
    return df