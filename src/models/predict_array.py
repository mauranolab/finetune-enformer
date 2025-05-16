#! /usr/bin/env python3

import os
import sys
import argparse

import tqdm
import pysam
import numpy as np
import tensorflow as tf

from dotenv import load_dotenv
from src.utils import pad_or_crop, dna_1hot
from src.models import load_model


def main(
        model:str, bedfile:str,
        reference:str, length:int,
        head: str, strides:int, output: str,
        key_size:int, value_size:int, num_heads:int) -> int:
    ## Check input
    for file in [bedfile, reference]:
        if not os.path.exists(file):
            print(f"! Unable to locate `{file}`", file=sys.stderr)
            return(1)
        
    if head not in ['mouse', 'human', 'finetune'] and not os.path.exists(file):
        print(f"! Unrecognized head `{head}`", file=sys.stderr)
        return(1)

    ## SETUP
    stride_it = [(i, i+length) for i in range(0, strides + 1)]
    reference = pysam.FastaFile(args.reference)

    ## Prepare track
    print("! Setup track", file=sys.stderr)
    total_length = length + strides
    targets = [line.rstrip().split("\t") for line in open(bedfile, "r")]
    targets = [(x[0], int(x[1]), int(x[2]))
               for x in targets if not x[0].startswith('#') and len(x) >= 3]
    targets = [(x[0], *pad_or_crop(x[1], x[2], total_length)) for x in targets]

    print("! Load model", file=sys.stderr)
    model = load_model(model, key_size, value_size, num_heads, head)
  
    ## Predict track signal
    print("! Predict tracks", file=sys.stderr)
    traget_prediction = []
    shape = None
    for (chrom, start, end) in tqdm.tqdm(targets):
        sequence = dna_1hot(reference.fetch(chrom, start, end).upper())
        sequence = np.array([sequence[st:ed] for st, ed in stride_it])
        try:
            prediction = []
            for bi in range(0, sequence.shape[0], 3):
                bs = sequence[bi:min(sequence.shape[0], bi+3), :, :]
                pd = model(bs, is_training=False)
                prediction.append(pd)
            prediction = np.concatenate(prediction)
            # prediction = model(sequence, is_training=False)
            prediction = np.mean(prediction, axis=0)
            shape = prediction.shape
            traget_prediction.append(prediction)
        except Exception as err:
            ## It will fail if none are present
            print(f"Unable to generate predictions at {chrom}:{start}-{end}", file=sys.stderr)
            print(err, file=sys.stderr)
            if shape is None:
                return(1)
            traget_prediction.append(np.full(shape, np.nan))            
    np.save(output, np.array(traget_prediction))    
    return(0)
    

if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(prog="predict-array")
    parser.add_argument('model', type=str)
    parser.add_argument('bedfile', type=str)
    parser.add_argument('--reference', type=str, default=os.getenv('MM10_FASTA'))
    parser.add_argument('--length', type=int, default=os.getenv('SEQ_LENGTH'))
    ## model parameters
    modeldef = parser.add_argument_group('model definition')
    modeldef.add_argument('--key-size', type=int, default=64)
    modeldef.add_argument('--value-size', type=int, default=64)
    modeldef.add_argument('--num-heads', type=int, default=1)
    ## output parameters
    outputdef = parser.add_argument_group('output definition')
    outputdef.add_argument('--head', type=str, default='mouse')
    outputdef.add_argument('--strides', type=int, default=10)
    outputdef.add_argument('--output', type=str, default="array.npy")

    args = parser.parse_args()
    print(args, file=sys.stderr)
    exit(main(**vars(args)))