import os
import sys
import argparse

import tqdm
import numpy as np
import matplotlib.pyplot as plt

from dotenv import load_dotenv


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


def main(target:str, contribution:str, prefix:str) -> int:
    if target != "-" and not os.path.exists(target):
        print(f"Unable to locate: `{target}`", file=sys.stderr)
        return(1)
    if not os.path.exists(contribution):
        print(f"Unable to locate: `{contribution}`", file=sys.stderr)
        return(1)
    
    print("! Loading target regions", file=sys.stderr)
    targets = sys.stdin if target == "-" else open(target, "r")
    targets = [__parse_bed12(line.rstrip().split("\t")) for line in targets]
    targets = sorted(targets)
    
    print("! Loading contribution", file=sys.stderr)
    contribution = np.load(contribution, allow_pickle=True)

    file = f"{prefix}.bedgraph"
    file = open(file, "w")

    for i in tqdm.tqdm(range(len(contribution))):
        chrom, start, end, sites = targets[i]
        # fig, axs = plt.subplots(len(sites), 1, sharex='col', figsize=(8, 0.75 * len(sites) + 0.25))
        # if len(sites) == 1:
        #     axs = [axs]
        for j in range(len(sites)):
            score = np.sum(contribution[i][j, :, :], axis=1)

            for k in range(score.shape[0]):
                print(chrom, start + k, start + k +1, score[k], sep="\t", file=file)
    file.close()
    return(0)


if __name__ == "__main__":
    load_dotenv()
    default_length = os.getenv("SEQ_LENGTH", 25600)

    parser = argparse.ArgumentParser(prog="contribution")
    parser.add_argument('target', type=str)
    parser.add_argument('contribution', type=str)
    parser.add_argument('--prefix', type=str, default="gradient")

    args = parser.parse_args()
    print(args, file=sys.stderr)
    exit(main(**vars(args)))