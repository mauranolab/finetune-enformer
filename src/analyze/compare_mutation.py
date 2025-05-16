import os
import sys
import argparse
from typing import List

import tqdm
import numpy as np
from matplotlib import pyplot as plt


def main(array:str, tracks:List[int], output:str) -> int:
    outstream = sys.stdout if output == "-" else open(output, "w")

    if not os.path.exists(array):
        print(f"Unable to locate: '{array}'", file=sys.stderr)
    array = np.load(array)

    ## Select central 512-640bp
    mid = array.shape[2] // 2
    st = mid - 2
    ed = mid + 2 + (array.shape[2] % 2)

    refarray = array[:, 0, st:ed, ...]
    altarray = array[:, 1, st:ed, ...]
    diffarray = altarray - refarray
    foldarray = np.log2(altarray / refarray)

    print("Index", "Track", "RefMax", "AltMax", "DiffMax", "Log2FCMax", sep="\t", file=outstream)
    for i in tqdm.tqdm(range(diffarray.shape[0])):
        diffi = np.argmax(np.abs(diffarray[i, :, tracks]), axis=-1)
        foldi = np.argmax(np.abs(foldarray[i, :, tracks]), axis=-1)
        for ji, j in enumerate(tracks):
            val = [
                np.max(refarray[i, :, j]),
                np.max(altarray[i, :, j]),
                diffarray[i, diffi[ji], j],
                foldarray[i, foldi[ji], j]]
            print(i, j, *np.round(val, 5), sep="\t", file=outstream)
        
    # if not plot:
    #     return(0) ## Early exit

    # if clip:
    #     refarray = np.clip(refarray, -clip, clip)
    #     altarray = np.clip(altarray, -clip, clip)
    
    # colsize = 10
    # rowsize = int(np.ceil(devarray.shape[0] / colsize))

    # for j in tracks:
    #     xaxs = np.arange(devarray.shape[1] * 128)

    #     fig, axs = plt.subplots(colsize, rowsize, figsize = (rowsize * 1.5, 8), sharex=True, sharey=True)
    #     for i in range(devarray.shape[0]):
    #         xsi = i // rowsize
    #         xsj = i % rowsize
    #         rvl = np.repeat(refarray[i, :, j], 128)
    #         avl = np.repeat(altarray[i, :, j], 128)
    #         axs[xsi, xsj].step(xaxs, rvl, label="ref", color="steelblue")
    #         axs[xsi, xsj].step(xaxs, avl, label="alt", color="firebrick")
    #         axs[xsi, xsj].annotate(i, xy=(.99, .99), xycoords='axes fraction', ha="right", va="top")
    #         if clip:
    #             axs[xsi, xsj].set_ylim([0, clip])
    #     axs[0, 0].legend(loc="upper left", fontsize="8")
    #     fig.tight_layout()
    #     fig.savefig(f"{prefix}.{j}.png")
    return(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="compare-mutation")
    parser.add_argument('array', type=str)
    parser.add_argument('tracks', nargs="+", type=int)
    parser.add_argument('--output', type=str, default='-')
    # parser.add_argument('--prefix', type=str, default="compare-mutation")
    # parser.add_argument('--plot', action='store_true')
    # parser.add_argument('--clip', type=int)

    args = parser.parse_args()
    print(args, file=sys.stderr)
    exit(main(**vars(args)))
