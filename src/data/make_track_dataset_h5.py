#! /usr/bin/env python3

import os
import sys
import csv
import argparse

import tqdm
import pysam
import pyBigWig
import h5py
import numpy as np

from dotenv import load_dotenv
from typing import Dict

from src.utils import dna_1hot

STAT = dict(
    sum  = lambda x: np.sum(x, axis=1, dtype=np.float32),
    mean = lambda x: np.mean(x, axis=1, dtype=np.float32)
)

class CovReader(object):
    def __init__(self, options:Dict):
        self._options = options
        self._covfile = pyBigWig.open(self._options["file"])
        if "clip" in self._options:
            self._clip = float(self._options["clip"])
        if "scale" in self._options:
            self._scale = float(self._options["scale"])
        self._func = STAT[self._options["sum_stat"]]
        
    def measure(self, chrom:str, start:int, end:int, width:int):
        cov = self._covfile.values(chrom, start, end, numpy=True)
        ## baseline to median
        baseline = np.percentile(cov, 50)
        baseline = np.nan_to_num(baseline)
        cov[np.isnan(cov)] = baseline

        cov = cov.reshape((end-start)//width, width)
        cov = self._func(cov)

        if self._scale is not None:
            cov = self._scale * cov
        if self._clip is not None:
            cov = np.clip(cov, -self._clip, self._clip)
        ## Clip to f16 limits
        cov = np.clip(cov, np.finfo(np.float16).min, np.finfo(np.float16).max)
        return(cov.astype(np.float16))

    def close(self):
        self._covfile.close()

    def __repr__(self):
        return(f"<CovReader {str(self._options)}>")


def main(mappable:str, tracks:str, reference:str, length:int, width:int, step:int, cutoff:float, size:int, batches:int, seed:int, prefix:str) -> int:
    ## Check files exists
    if mappable != "-" and not os.path.exists(mappable):
        print(f"Unable to locate file at`{mappable}`", file=sys.stderr)
        return(1)
    if not os.path.exists(tracks):
        print(f"Unable to locate file at`{tracks}`", file=sys.stderr)
        return(1)

    ## Setup
    np.random.seed(seed)
    reference = pysam.Fastafile(reference)
    total_size = batches * size

    ## Read and prepare targets
    tracks = csv.DictReader(open(tracks, "r"), delimiter="\t")
    tracks = [CovReader(line) for line in tracks]

    ## Read mappable regions
    chrom_ix = []
    start_pool = []

    mappable = sys.stdin if mappable == "-" else open(mappable, "r")
    for line in tqdm.tqdm(mappable):
        chrom, start, end = line.rstrip().split("\t")[:3]
        start = int(start)
        end = int(end)

        if end - start < length:
            continue

        if chrom not in chrom_ix:
            chrom_ix.append(chrom)

        st = np.arange(start, end - length, step=step)
        ch = np.repeat(chrom_ix.index(chrom), st.shape[0])
        start_pool.append(np.stack((ch, st), axis=-1))
    start_pool = np.concatenate(start_pool)

    ## Sample sites
    if total_size > start_pool.shape[0]:
        total_size = start_pool.shape[0]
    ## Shuffle entries
    ix = np.random.choice(np.arange(start_pool.shape[0]), size=total_size)
    np.random.shuffle(ix)
    sites = start_pool[ix, :]

    dataset = open(f"{prefix}.bed", "w")

    counter = 0
    index = 0

    for batch in tqdm.tqdm(range(batches)):
        if counter >= total_size:
            break ## Early exit

        input = []
        outcome = []

        for ch, start in sites[index:, ...]:
            index += 1
            chrom = chrom_ix[ch]
            end = start + length

            out = [track.measure(chrom, start, end, width) for track in tracks]
            out = np.stack(out, axis=-1)

            if cutoff and np.max(np.std(out, 0)) < cutoff:
                continue ## SKIP if not informative
        
            seq = reference.fetch(chrom, start, end).upper()
            seq = dna_1hot(seq, dtype=np.float16)

            input.append(seq)
            outcome.append(out)
            print(chrom, start, end, sep="\t", file=dataset)

            counter += 1
            if counter > 0 and counter % size == 0 or counter >= total_size:
                break
        
        with h5py.File(f"{prefix}.{batch}.h5", 'w') as h5:
            h5.create_dataset('input', data=np.array(input), dtype='float16', compression="gzip")
            h5.create_dataset('outcome', data=np.array(outcome), dtype='float16', compression="gzip")

    dataset.close()
    for track in tracks:
        track.close()
    
    return(0)


if __name__ == "__main__":
    load_dotenv()
    
    parser = argparse.ArgumentParser(prog="make-track-dataset")
    parser.add_argument('mappable', type=str)
    parser.add_argument('tracks', type=str)
    parser.add_argument('--reference', type=str, default=os.getenv("MM10_FASTA"))
    parser.add_argument('--length', type=int, default=int(os.getenv("SEQ_LENGTH")))
    parser.add_argument('--width', type=int, default=int(os.getenv("SEQ_WINDOW")))
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--size', type=int, default=10000)
    parser.add_argument('--batches', type=int, default=5)
    parser.add_argument('--cutoff', type=float)
    parser.add_argument('--seed', type=int, default=5)
    parser.add_argument('--prefix', type=str, default="track-dataset")

    args = parser.parse_args()

    print(args, file=sys.stderr)
    exit(main(**vars(args)))