#! /usr/bin/env python3

import os
import sys
import argparse

import tqdm
import h5py
import pysam
import numpy as np
import tensorflow as tf

from dotenv import load_dotenv
from src.utils import pad_or_crop, dna_1hot
from src.models import load_model, load_model_from_modeldef, head_func


def main(model:str, bedfile:str, reference:str, length:int,
         head: str, strides:int, output: str,
         key_size:int, value_size:int, num_heads:int) -> int:
    ## Check input
    for file in [bedfile, reference]:
        if not os.path.exists(file):
            print(f"! Unable to locate `{file}`", file=sys.stderr)
            return(1)
    
    ## SETUP
    stride_it = [(i, i+length) for i in range(0, strides + 1)]
    reference = pysam.FastaFile(args.reference)

    ## Load model
    print("! Load model", file=sys.stderr)
    mdict = load_model_from_modeldef(model)
    model = head_func(mdict, head) ## Accept a custom head

    ## Prepare track
    print("! Setup track", file=sys.stderr)
    total_length = length + strides
    targets = [line.rstrip().split("\t") for line in open(bedfile, "r")]
    targets = [(x[0], int(x[1]), int(x[2]))
               for x in targets if not x[0].startswith('#') and len(x) >= 3]
    targets = [(x[0], *pad_or_crop(x[1], x[2], total_length)) for x in targets]
    targets = [
        (chrom, start, end, reference.fetch(chrom, start, end).upper())
        for chrom, start, end in targets
    ]

    ## Predict track signal
    print("! Predict tracks", file=sys.stderr)
    standard_shape = None
    predictons = []
    for (chrom, start, end, sequence) in tqdm.tqdm(targets):
        sequence = dna_1hot(sequence)
        sequence = np.array([sequence[st:ed] for st, ed in stride_it])

        try:
            prediction = model(sequence, is_training=False)
            prediction = np.mean(prediction, axis=0)
            standard_shape = prediction.shape
            predictons.append(prediction)
        except Exception as err:
            print(f"Unable to generate predictions at {chrom}:{start}-{end}", file=sys.stderr)
            print(err, file=sys.stderr)
            ## Unless it fails the first interaction, it will fill with an empty vector
            if standard_shape is None:
                return(1)
            predictons.append(np.full(standard_shape, np.nan))
    ## Save as hdf5 file
    with h5py.File(output, 'w') as h5:
        chroms = list(set(chrom for chrom, _, _, _ in targets))
        chroms_map = dict((chrom, i) for i, chrom in enumerate(chroms))
        chroms_idx = np.array([chroms_map[c] for c, _, _, _ in targets], dtype='uint8')
        starts = np.array([start for _, start, _, _ in targets], dtype='uint32')
        ends = np.array([end for _, _, end, _ in targets], dtype='uint32')
        outcome = np.array(predictons, dtype='float16')

        h5.create_dataset('chrom_label', data=chroms, dtype=h5py.string_dtype(encoding='utf-8'))
        h5.create_dataset('chrom', data=chroms_idx, dtype='uint8')
        h5.create_dataset('start', data=starts, dtype='uint32', compression='gzip')
        h5.create_dataset('end', data=ends, dtype='uint32', compression='gzip')
        h5.create_dataset('prediction', data=outcome, dtype='float16', compression='gzip')
    return(0)
    

if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(prog="predict-array")
    parser.add_argument('model', type=str)
    parser.add_argument('bedfile', type=str)
    parser.add_argument('--reference', type=str)
    parser.add_argument('--length', type=int, default=os.getenv('SEQ_LENGTH', 25600))
    ## model parameters
    modeldef = parser.add_argument_group('model definition')
    modeldef.add_argument('--key-size', type=int, default=64)
    modeldef.add_argument('--value-size', type=int, default=64)
    modeldef.add_argument('--num-heads', type=int, default=1)
    ## output parameters
    outputdef = parser.add_argument_group('output definition')
    outputdef.add_argument('--head', type=str, default='mouse')
    outputdef.add_argument('--strides', type=int, default=10)
    outputdef.add_argument('--output', type=str, default="array.h5")

    args = parser.parse_args()
    print(args, file=sys.stderr)
    exit(main(**vars(args)))