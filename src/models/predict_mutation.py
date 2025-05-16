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


def parse_coordinate(coordinate:str) -> tuple:
    chrom, coord, *anno = coordinate.split(":")
    start, end = coord.split("-")
    return (chrom, int(start), int(end), anno)


def parse_payload(name, coord, length, context_sequence, reference):
    sequences = []
    for c in coord.split(","):
        chrom, start, end, anno = parse_coordinate(c)
        anno = [a for a in anno if a.startswith("f")]
        if len(anno) > 0:
            ref = pysam.FastaFile(anno[0][1:])
        else:
            ref = reference
        sequences.append(ref.fetch(chrom, start, end))
    payload_sequence = "".join(sequences)
    sequence = context_sequence[0] + payload_sequence + context_sequence[1]
    (cropstart, cropend) = pad_or_crop(0, len(sequence), length)
    sequence = sequence[cropstart:cropend]
    offset = len(context_sequence[0]) - cropstart

    return(name, sequence, (offset, offset + len(payload_sequence)))


def __parse_mutation(entry):
    return(dict(
        chrom = entry[0],
        position = int(entry[1]),
        rsid = entry[3],
        ref = entry[4].upper(),
        alt = entry[6].upper()))


def __add_sequence(mutation, reference, length):
    padding = length // 2
    start = mutation["position"] - padding

    sequence = reference.fetch(mutation["chrom"], start, start + length).upper()
    if mutation["ref"] != sequence[padding]:
        print(f"! Reference allele doesn't match genome allele", file=sys.stderr)

    mutation["refseq"] = sequence[:padding] + mutation["ref"] + sequence[(padding+1):]
    mutation["altseq"] = sequence[:padding] + mutation["alt"] + sequence[(padding+1):]

    return(mutation)


def __stride_sequence(sequence, stride_it):
    sequence = dna_1hot(sequence)
    sequence = np.array([sequence[st:ed] for st, ed in stride_it])
    return(tf.convert_to_tensor(sequence, np.float32))


def __crop(vector, length):
    trim = (vector.shape[-2] - length) // 2
    return vector[..., trim:-trim, :]


def main(model:str, mutations:str, length:int, reference:str,
        head: str, strides:int, output: str,
        key_size:int, value_size:int, num_heads:int) -> int:
    ## Validate input
    if not os.path.exists(reference):
        print(f"! Unable to locate reference file at: `{reference}`", file=sys.stderr)
        return(1)
    
    if mutations != "-" and not os.path.exists(mutations):
        print(f"! Unable to locate mutations file at: `{mutations}`", file=sys.stderr)
        return(1)
    
    ## Setup
    print("! Load model", file=sys.stderr)
    model = load_model(model, key_size, value_size, num_heads, head)

    reference = pysam.FastaFile(args.reference)
    total_length = length + strides
    stride_it = [(i, i + length) for i in range(0, strides + 1)]

    mutations = sys.stdin if mutations == "-" else open(mutations, 'r')
    mutations = [line.rstrip().split("\t") for line in mutations]
    mutations = [__parse_mutation(entry) for entry in mutations[1:]] # Skip header line
    mutations = [__add_sequence(mutation, reference, total_length) for mutation in mutations]

    ## Predict track signal
    print("! Predict tracks", file=sys.stderr)
    results = []
    for mutation in tqdm.tqdm(mutations):
        refseq = __stride_sequence(mutation["refseq"], stride_it)
        altseq = __stride_sequence(mutation["altseq"], stride_it)

        ref_prediction = model(refseq, is_training=False)
        ref_prediction = tf.math.reduce_mean(__crop(ref_prediction, 10), axis=0)
        alt_prediction = model(altseq, is_training=False)
        alt_prediction = tf.math.reduce_mean(__crop(alt_prediction, 10), axis=0)

        prediction = tf.stack((ref_prediction, alt_prediction))
        print(prediction)

        results.append(prediction)
    
    results = np.array([tensor.numpy() for tensor in results])
    np.save(output, results)

    return(0)


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(prog="predict-mutation")
    parser.add_argument('model', type=str)
    parser.add_argument('mutations', type=str)
    parser.add_argument('--length', type=int, default=os.getenv('SEQ_LENGTH', 25600))
    parser.add_argument('--reference', type=str)
    ## model parameters
    modeldef = parser.add_argument_group('model definition')
    modeldef.add_argument('--key-size', type=int, default=64)
    modeldef.add_argument('--value-size', type=int, default=64)
    modeldef.add_argument('--num-heads', type=int, default=1)
    ## output parameters
    outputdef = parser.add_argument_group('output definition')
    outputdef.add_argument('--head', type=str, choices=['mouse', 'human'], default='mouse')
    outputdef.add_argument('--strides', type=int, default=10)
    outputdef.add_argument('--output', type=str, default="array.npy")

    args = parser.parse_args()
    print(args, file=sys.stderr)
    exit(main(**vars(args)))