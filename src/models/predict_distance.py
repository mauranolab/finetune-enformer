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
from src.models import load_model


def shuffle_dinuc(seq:str, n:int = 1, seed:int = None) -> np.array:
    if seed:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()

    arr = np.frombuffer(bytearray(seq, "utf8"), dtype=np.int8)
    chars, tokens = np.unique(arr, return_inverse=True)
    shuf_next_inds = [ np.where(tokens[:-1] == t)[0] + 1 for t in range(len(chars)) ]

    results = []
    for i in range(n):
        # shuffle next indices
        for t in range(len(chars)):
            inds = np.arange(len(shuf_next_inds[t]))
            inds[:-1] = rng.permutation(len(inds) - 1)
            shuf_next_inds[t] = shuf_next_inds[t][inds]

        # Build the resulting array
        counters = [0] * len(chars)

        ind = 0
        result = np.empty_like(tokens)
        result[0] = tokens[ind]
        for j in range(1, len(tokens)):
            t = tokens[ind]
            ind = shuf_next_inds[t][counters[t]]
            counters[t] += 1
            result[j] = tokens[ind]
        results.append(chars[result].tobytes().decode("ascii"))
    return(results)


def dinuc_shuffle(seq:np.array, n:int = 1, seed:int = None) -> np.array:
    if seed:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    
    length = seq.shape[0]
    result = np.empty_like(seq, shape=(n, length, 4))
    for i in range(n):
        p = rng.permutation(length // 2) * 2
        p = np.reshape(np.array([p, p+1]), -1, order="F")
        result[i, :, :] = seq[p, :]
    return(result)


def parse_coord(coord:str) -> tuple:
    name, location = coord.split(":")
    start, end = location.split("-")
    return(name, int(start), int(end))


def main(model:str, sites:str, length:int, step_size:int, reference:str,
        head: str, shuffles:int, output:str, report:str,
        key_size:int, value_size:int, num_heads:int, seed:int) -> int:
    ## Check required files
    if not os.path.isfile(sites):
        print(f"Unable to locate `{sites}`", file=sys.stderr)
        exit(1)
    if not os.path.isfile(reference):
        print(f"Unable to locate `{reference}`", file=sys.stderr)
        exit(1)

    ## SETUP
    print("! Setup", file=sys.stderr)
    reference = pysam.FastaFile(args.reference)
    reportfile = sys.stdout if report == "-" else open(report, "w")

    ## Build virtual sequences
    sitedat = [line.strip().split("\t") for line in open(sites, 'r')]
    sitedat = [
        (x[0], int(x[1]), int(x[2]), parse_coord(x[3]), parse_coord(x[4]))
        for x in sitedat]
    
    print("! Load model", file=sys.stderr)
    model = load_model(model, key_size, value_size, num_heads, head)
    

    print("! Predicting effect", file=sys.stderr)
    for chrom, start, end, fixed, mobile in tqdm.tqdm(sitedat):
        pstart, pend = pad_or_crop(start, end, length)
        offset = pstart - start

        seq = reference.fetch(chrom, pstart, pend)
        shufseq = shuffle_dinuc(seq, shuffles, seed)
        seq = dna_1hot(seq)

        ## Extract fixed and mobile
        fixedStart = (fixed[1] + offset)
        fixedEnd = (fixed[2] + offset)
        fixedSeq = seq[fixedStart:fixedEnd, :]

        mobileStart = (mobile[1] + offset)
        mobileEnd = (mobile[2] + offset)
        mobileSeq = seq[mobileStart:mobileEnd, :]
        mobileLen = mobileSeq.shape[0]

        ## Shuffle sequence and restore fixed
        shufseq = np.array([dna_1hot(s) for s in shufseq])
        shufseq[:, fixedStart:fixedEnd, :] = fixedSeq

        ## Build steps
        steps = np.concatenate([
            np.arange(fixedStart % step_size, fixedStart, step_size),
            np.arange(fixedEnd, length, step_size)])
        
        for stepStart in tqdm.tqdm(steps, leave=False):
            stepEnd = stepStart + mobileLen
            distance = stepStart-fixedStart
            if stepEnd > length:
                continue
            stepSeq = shufseq.copy()
            stepSeq[:, stepStart:stepEnd, :] = mobileSeq

            pred = model(stepSeq, is_training=False)
            pred = tf.math.reduce_mean(pred, axis=0) ## Average across shuffles
            pred = tf.repeat(pred, 128, 0) ## Expand to bp resolution
            
            ## Select central 256bp
            centralPoint = pred.shape[0] // 2
            fixedAvg  = tf.math.reduce_mean(pred[(centralPoint-128):(centralPoint+128), :], 0).numpy()
            mobileMax = tf.math.reduce_max(pred[stepStart:stepEnd, :], 0).numpy()

            print(chrom, start, end, distance, mobile[1], mobile[2], fixed[0], *fixedAvg.tolist(), sep="\t", file=reportfile)
            print(chrom, start, end, distance, mobile[1], mobile[2], mobile[0], *mobileMax.tolist(), sep="\t", file=reportfile)
            # result.append(dict(
            #     chrom=chrom, location=(start, end), distance=distance,
            #     fixed=fixed, fixedPred=fixedPred,
            #     mobile=mobile, mobilePred=mobilePred))
    
    # print("! Saving predictions", file=sys.stderr)
    # ## Save as hdf5 file
    # with h5py.File(output, 'w') as h5:
    #     chroms = list(set(x["chrom"] for x in result)).sort
    #     chroms_map = dict((chrom, i) for i, chrom in enumerate(chroms))
    #     chroms_idx = np.array([chroms_map[x["chrom"]] for x in result], dtype='uint8')
    #     starts = np.array([x["location"][0] for x in result], dtype='uint32')
    #     ends = np.array([x["location"][1] for x in result], dtype='uint32')
    #     distance = np.array([x["distance"] for x in result], dtype='int32')
    #     mobileStart = np.array([x["mobile"][1] for x in result], dtype='uint32')
    #     mobileEnd = np.array([x["mobile"][2] for x in result], dtype='uint32')
    #     fixedPrediction = np.array([x["fixedPred"] for x in result], dtype='float16')
    #     mobilePrediction = np.array([x["mobilePred"] for x in result], dtype='float16')

    #     h5.create_dataset('chrom_label', data=chroms, dtype=h5py.string_dtype(encoding='utf-8'))
    #     h5.create_dataset('chrom', data=chroms_idx, dtype='uint8')
    #     h5.create_dataset('start', data=starts, dtype='uint32', compression='gzip')
    #     h5.create_dataset('end', data=ends, dtype='uint32', compression='gzip')
    #     h5.create_dataset('mobileStart', data=mobileStart, dtype='uint32', compression='gzip')
    #     h5.create_dataset('mobileEnd', data=mobileEnd, dtype='uint32', compression='gzip')
    #     h5.create_dataset('distance', data=distance, dtype='int32', compression='gzip')
    #     h5.create_dataset('fixedPrediction', data=fixedPrediction, dtype='float16', compression='gzip')
    #     h5.create_dataset('mobilePrediction', data=mobilePrediction, dtype='float16', compression='gzip')  
    
    if report != "-":
        reportfile.close()
    return(0)


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(prog="predict-distance")
    parser.add_argument('model', type=str)
    parser.add_argument('sites', type=str)
    parser.add_argument('--length', type=int, default=os.getenv('SEQ_LENGTH', 25600))
    parser.add_argument('--step-size', type=int, default=10000)
    parser.add_argument('--reference', type=str)
    parser.add_argument('--seed', type=int, default=42)
    ## model parameters
    modeldef = parser.add_argument_group('model definition')
    modeldef.add_argument('--key-size', type=int, default=64)
    modeldef.add_argument('--value-size', type=int, default=64)
    modeldef.add_argument('--num-heads', type=int, default=1)
    ## output parameters
    outputdef = parser.add_argument_group('output definition')
    outputdef.add_argument('--head', type=str, choices=['mouse', 'human'], default='mouse')
    outputdef.add_argument('--shuffles', type=int, default=5)
    outputdef.add_argument('--output', type=str, default="array.h5")
    outputdef.add_argument('--report', type=str, default="-")

    args = parser.parse_args()
    print(args, file=sys.stderr)
    exit(main(**vars(args)))