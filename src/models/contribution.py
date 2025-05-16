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


def __parse_bed12(entry:list) -> tuple:
    chrom = entry[0]
    start = int(entry[1])
    end = int(entry[2])

    sites = []
    if len(entry) >= 12:
        bSizes = [int(x) for x in entry[10].split(",")]
        bStart = [int(x) for x in entry[11].split(",")]
        for i in range(len(bSizes)):
            sites.append((bStart[i], bStart[i] + bSizes[i]))
    return((chrom, start, end, sites))


def wrap(vec: np.ndarray, size:int) -> np.ndarray:
    nr, nc = vec.shape
    return(np.reshape(vec, (nr, nc // size, size)).max(axis=2))


def __contrib_gradient(model, input:np.ndarray, mask:np.ndarray, track:int):
    input = input[tf.newaxis]

    mass = tf.reduce_sum(mask, 1)
    grad = []
    for i in range(mask.shape[0]):
        with tf.GradientTape() as tape:
            tape.watch(input)
            pred = model(input, is_training=False)
            pred = tf.reduce_sum(mask[tf.newaxis, i, :] * pred[:, :, track]) / mass[tf.newaxis, i]
        grad.append(tape.gradient(pred, input) * input)
    grad = tf.concat(grad, axis=0)
    return(grad)


def main(model:str, target:str, reference:str, length:int,
         key_size:int, value_size:int, num_heads:int,
         head:str, track:int, output:str) -> int:
    if not os.path.exists(reference):
        print(f"Unable to locate: `{reference}`", file=sys.stderr)
        return(1)
    if target != "-" and not os.path.exists(target):
        print(f"Unable to locate: `{target}`", file=sys.stderr)
        return(1)
    
    print("! Loading reference genome", file=sys.stderr)
    reference = pysam.FastaFile(args.reference)

    ## load target as bed12
    print("! Loading target regions", file=sys.stderr)
    targets = sys.stdin if target == "-" else open(target, "r")
    targets = [__parse_bed12(line.rstrip().split("\t")) for line in targets]
    
    print("! Loading enformer model", file=sys.stderr)
    model = load_model(model, key_size, value_size, num_heads, head)

    print("! Predicting target contribution", file=sys.stderr)
    grads = []
    for chrom, start, end, sites in tqdm.tqdm(targets):
        start, end = pad_or_crop(start, end, length)
        sequence = reference.fetch(chrom, start, end).upper()
        sequence = dna_1hot(sequence)
        sequence = tf.convert_to_tensor(sequence, dtype=tf.float32)

        mask = np.zeros((len(sites), length), dtype=np.float32)
        for i, (bstart, bend) in enumerate(sites):
            mask[i, bstart:bend] = 1
        mask = wrap(mask, 128)
        mask = tf.convert_to_tensor(mask, dtype=tf.float32)

        grad = __contrib_gradient(model, sequence, mask, track)
        grads.append(grad.numpy())
    np.save(output, np.array(grads, dtype=object), allow_pickle=True)
    return(0)


if __name__ == "__main__":
    load_dotenv()
    default_length = os.getenv("SEQ_LENGTH", 25600)

    parser = argparse.ArgumentParser(prog="contribution")
    parser.add_argument('model', type=str)
    parser.add_argument('target', type=str)
    parser.add_argument('--reference', type=str, default=os.getenv('MM10_FASTA'))
    parser.add_argument('--length', type=int, default=os.getenv('SEQ_LENGTH'))
    ## model parameters
    modeldef = parser.add_argument_group('model definition')
    modeldef.add_argument('--key-size', type=int, default=64)
    modeldef.add_argument('--value-size', type=int, default=64)
    modeldef.add_argument('--num-heads', type=int, default=1)
    ## output parameters
    outputdef = parser.add_argument_group('output definition')
    outputdef.add_argument('--head', type=str, choices=['mouse', 'human'], default='mouse')
    outputdef.add_argument('--track', type=int, default=10)
    outputdef.add_argument('--output', type=str, default="array.npy")

    args = parser.parse_args()
    print(args, file=sys.stderr)
    exit(main(**vars(args)))