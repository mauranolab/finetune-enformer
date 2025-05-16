import os
import sys
import argparse
from collections import defaultdict
from typing import List, Dict, Iterable, Callable

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def is_number(element:str) -> bool:
    partition=element.partition('.')
    if ((partition[0].isdigit() and partition[1]=='.' and partition[2].isdigit()) or
        (partition[0]=='' and partition[1]=='.' and partition[2].isdigit()) or
        (partition[0].isdigit() and partition[1]=='.' and partition[2]=='') or
        (partition[0].isdigit() and partition[1]=='' and partition[2]=='')):
        return True
    else:
        return False


def read_score(score:Iterable[str]) -> Dict[str, np.ndarray]:
    header = next(score)
    header = header.rstrip().split("\t")
    data = dict((name, []) for name in header)

    for entry in score:
        entry = entry.rstrip().split("\t")
        for i in range(len(header)):
            value = entry[i] if i in [0, 3] else float(entry[i])
            data[header[i]].append(value)
    
    for i in range(len(header)):
        data[header[i]] = np.array(data[header[i]])

    return(data)


def ecdf(x: np.ndarray) -> Callable[[np.array], np.array]:
    ## See https://stackoverflow.com/questions/15792552/numpy-scipy-equivalent-of-r-ecdfxx-function
    xs = np.sort(x)
    n = xs.shape[0]
    def _ecdf(v: np.ndarray) -> np.array:
        return np.searchsorted(xs, v, side='right') / n
    return _ecdf


def beeswarm(y:np.ndarray, width:float = 0.8) -> np.ndarray:
    """
    see https://stackoverflow.com/questions/36153410/how-to-create-a-swarm-plot-with-matplotlib
    """
    y = np.asarray(y)
    x = np.zeros(len(y))
    # Get upper bounds of bins
    nb = len(y) // 6
    ylo = np.min(y)
    yhi = np.max(y)
    dy = (yhi-ylo) / nb
    yb = np.linspace(ylo+dy, yhi-dy, nb-1)
    ## divide into bins
    i = np.arange(len(y))
    ibs = [0] * nb
    ybs = [0] * nb
    nmx = 0
    for j, ybin in enumerate(yb):
        f = y <= ybin
        ibs[j], ybs[j] = i[f], y[f]
        nmx = max(nmx, len(ibs[j]))
        f = ~ f
        i, y = i[f], y[f]
    ibs[-1], ybs[-1] = i, y
    nmx = max(nmx, len(ibs[-1]))
    # Assign x indices
    dx = 1 / (nmx // 2) * width / 2
    for i, y in zip(ibs, ybs):
        if len(i) > 1:
            j = len(i) % 2
            i = i[np.argsort(y)]
            a = i[j::2]
            b = i[j+1::2]
            x[a] = (0.5 + j / 3 + np.arange(len(b))) * dx
            x[b] = (0.5 + j / 3 + np.arange(len(b))) * -dx
    return x


def main(score:str, prefix:str, length:int, top:int) -> int:
    if not os.path.exists(score):
        print(f"~ Unable to locate score table at `{score}`", file=sys.stderr)
        return(1)
    
    ## SETUP
    mpl.rcParams['font.size'] = 8

    ## Load data
    data = read_score(open(score, "r"))
    cutoffs = [0.5, 0.7, 0.8, 0.9]

    tracks = [key for key in data.keys() if key not in ['chrom', 'start', 'end', 'name']]
    track_colors = ["#29AAE1"] + ["#6a65aa"] * (len(tracks) - 5) + ["#999999", "#E69F00", "#CC79A7", "#009E73"]
    reference = tracks[0]

    ## CDF
    minval = min(np.min(data[x]) for x in tracks)
    maxval = max(np.max(data[x]) for x in tracks)
    valspace = np.linspace(minval, maxval, 1000)
    # valspace = np.linspace(np.log10(minval + 1), np.log10(maxval + 1), 1000)
    # valspace = (10 ** valspace) - 1

    fig, ax = plt.subplots(figsize=(2.75, 2.5))
    for i, track in enumerate(tracks):
        cdf = ecdf(data[track])(valspace)
        ax.step(valspace, cdf, color=track_colors[i], label=tracks[i])
    
    ax.hlines(cutoffs, xmin=minval, xmax=maxval, colors='gray', linestyles='dotted')
    ax.legend()
    ax.set_xscale("log")
    ax.set_yticks([0, 0.2] + cutoffs)
    
    fig.tight_layout()
    fig.savefig(f"{prefix}.cdf.png", dpi=150)
    fig.savefig(f"{prefix}.cdf.pdf")

    ## Correlation
    chroms = np.unique(data['chrom'])
    corr = dict((track, []) for track in tracks)
    for chrom in chroms:
        cvec = data['chrom'] == chrom
        refvalue = data[reference][cvec]
        for track in tracks:
            value = data[track][cvec]
            if np.std(value) == 0 or np.std(refvalue) == 0:
                rcoef = np.nan
            else:
                rcoef = np.corrcoef(value, refvalue)[0,1]
            corr[track].append(rcoef)
    
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    for i, track in enumerate(tracks):
        rcoef = corr[track]
        mean = np.mean(rcoef)
        std = np.std(rcoef)
        ax.scatter(beeswarm(rcoef) + i, rcoef, label=track[i], alpha=0.5,
                   facecolor='none', edgecolor=track_colors[i])
        ax.scatter(i, mean, label=track[i], color=track_colors[i])
        ax.errorbar(i, mean, yerr=std, label=track[i], color=track_colors[i])
        if track == "Enformer":
            ax.hlines(mean, -0.5, len(tracks) - 0.5, linestyle='dotted', colors='gray')
    
    ax.set_ylabel('Pearson r')
    fig.tight_layout()
    fig.savefig(f"{prefix}.corr.png", dpi=150)
    fig.savefig(f"{prefix}.corr.pdf")

    ## Call replication
    call = lambda x, q: x >= np.quantile(x, q)
    track_call = dict(
        (cutoff, dict((t, call(data[t], cutoff)) for t in tracks))
        for cutoff in cutoffs
    )
    perf = []
    for cutoff in track_call:
        for chrom in chroms:
            cvec = data['chrom'] == chrom
            refvalue = track_call[cutoff][reference][cvec]
            for track in tracks:
                value = track_call[cutoff][track][cvec]
                perf.append(dict(
                    track=track, chrom=chrom, cutoff=cutoff,
                    accuracy = np.mean(refvalue == value),
                    true_positive = np.mean(value[refvalue]),
                    true_negative = np.mean(~value[~refvalue])
                ))

    fig, ax = plt.subplots(1, 3, figsize=(8, 3), sharey=True)
    for i, measure in enumerate(['accuracy', 'true_positive', 'true_negative']):
        ax[i].set_ylabel(measure)
        ax[i].set_xlabel('Quantile cutoff')
        ax[i].set_xticks(np.arange(len(cutoffs)), cutoffs)
        ax[i].set_box_aspect(1)
        for j, t in enumerate(tracks):
            value = [
                [p[measure] for p in perf if p['track'] == t and p['cutoff'] == c]
                for c in cutoffs
            ]
            mean = [np.mean(v) for v in value]
            std = [np.std(v) for v in value]

            ax[i].scatter(np.arange(len(cutoffs)), mean, color=track_colors[j], label=t)
            ax[i].errorbar(np.arange(len(cutoffs)), mean, yerr=std, color=track_colors[j])
    ax[2].legend()
    
    fig.tight_layout()
    fig.savefig(f"{prefix}.call.png", dpi=150)
    fig.savefig(f"{prefix}.call.pdf")

    return(0)

if __name__ == "__main__":
    default_length = os.getenv("SEQ_LENGTH", 25600)

    parser = argparse.ArgumentParser("analyze.peak")
    parser.add_argument('score', type=str)
    parser.add_argument('--prefix', type=str, default="peak")
    parser.add_argument('--length', type=int, default=default_length)
    parser.add_argument('--top', type=int, default=20)

    args = parser.parse_args()
    print(args, file=sys.stderr)
    exit(main(**vars(args)))