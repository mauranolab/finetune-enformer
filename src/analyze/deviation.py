import os
import sys
import argparse
from collections import defaultdict
from typing import List, Dict, Tuple, Iterable, Callable

import tqdm
import pyBigWig
import numpy as np
from intervaltree import IntervalTree


def main(block:str, array:str, track:str, index:int, output:str) -> int:
    if block != "-" and not os.path.exists(block):
        print(f"! Unable to locate `{block}`", file=sys.stderr)
        return(1)
    
    if not os.path.exists(array):
        print(f"! Unable to locate `{array}`", file=sys.stderr)
        return(1)
    
    if not os.path.exists(track):
        print(f"! Unable to locate `{track}`", file=sys.stderr)
        return(1)
    step = 128

    print("! Loading prediction array", file=sys.stderr)
    array = np.load(array)[:, :, index]

    print("! Loading coverage track", file=sys.stderr)
    track = pyBigWig.open(track)

    print("! Loading targets", file=sys.stderr)
    block = sys.stdin if block == "-" else open(block, "r")
    blocks = [line.rstrip().split() for line in block]
    blocks = [(line[0], int(line[1]), int(line[2])) for line in blocks]

    print("! Calculating deviation", file=sys.stderr)
    values = []
    for chrom, start, end in tqdm.tqdm(blocks):
        val = track.values(chrom, start, end, numpy=True)
        val = np.nan_to_num(val)
        val = np.reshape(val, (val.shape[0] // step, step))
        val = np.mean(val, axis=1)
        values.append(val)
    values = np.array(values)

    ## regularize
    # x = np.reshape(array, (array.shape[0] * array.shape[1]))
    # y = np.reshape(values, (values.shape[0] * values.shape[1]))
    # sample = np.random.choice(array.shape[0], int(array.shape[0] / 2), replace=False)
    # b, m = np.polyfit(array[sample,:].flatten(), values[sample,:].flatten(), 1)
    b, m = np.polyfit(array.flatten(), values.flatten(), 1)
    
    ## pearson residual
    resid = values - (m + b * array)
    # resid = values - array
    std = np.std(resid.flatten())
    resid = resid / np.std(resid.flatten())
    resid = np.repeat(resid, step, axis = 1)

    values = np.repeat(values, step, axis = 1)
    array = np.repeat(array, step, axis = 1)

    print("! Generate bedgraph", file=sys.stderr)
    outfile = sys.stdout if output == "-" else open(output, "w")
    
    citree = defaultdict(IntervalTree)
    fitree = defaultdict(IntervalTree)
    for i, (chrom, start, end) in tqdm.tqdm(enumerate(blocks)):
        citree[chrom].addi(start, end, i)
        fitree[chrom].addi(start, end)
    
    for chrom in citree:
        itree = citree[chrom]
        ftree = fitree[chrom]
        ftree.split_overlaps()

        bedgraph = []
        for it in ftree:
            size = it.end - it.begin
            vals = np.zeros((it.end - it.begin,))
            x = np.zeros((it.end - it.begin,))
            y = np.zeros((it.end - it.begin,))

            overlaps = itree.overlap(it)
            num_ovlp = len(overlaps)
            for overlap in overlaps:
                offset = it.begin - overlap.begin
                vals += resid[overlap.data, offset:(offset + size)] / num_ovlp
                x += array[overlap.data, offset:(offset + size)] / num_ovlp
                y += values[overlap.data, offset:(offset + size)]  / num_ovlp
            vals = np.round(vals, decimals=5)

            ## Merge continuous runs
            is_close = np.isclose(vals[1:], vals[:-1], equal_nan=True)
            position = np.r_[0, np.flatnonzero(~ is_close) + 1]
            length = np.diff(np.r_[position, vals.shape[0]])

            for i in range(position.shape[0]):
                pi = position[i]
                start = it.begin + pi
                end = it.begin + pi + length[i]
                bedgraph.append((chrom, start, end, vals[pi])) #, x[pi], y[pi], m, b, std))
        
        bedgraph.sort()
        for entry in bedgraph:
            if entry[2] <= entry[1]:
                print("ERROR", *entry, sep="\t")
                return(0)
            print(*entry, sep="\t", file=outfile)
    return(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("analyze.deviation")
    parser.add_argument('block', type=str)
    parser.add_argument('array', type=str)
    parser.add_argument('track', type=str)
    parser.add_argument('--index', type=int, default=10)
    parser.add_argument('--output', type=str, default="-")

    args = parser.parse_args()
    print(args, file=sys.stderr)
    exit(main(**vars(args)))