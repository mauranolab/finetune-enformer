import numpy as np

from src.utils import fold_split

def main() -> int:
    ratio = [1, 1, 1, 1]
    sample = 1000
    size = 72

    folds = [fold_split(ratio, size) for _ in range(sample)]
    folds = np.array(folds)

    counts = [np.bincount(folds[:, i]) for i in range(size)]
    counts = np.array(counts)
    
    for i in range(len(ratio)):
        print(f"fold{i}", *counts[:, i] / sample, sep="\t")
    print()
    print(*np.mean(counts / sample, 0), sep="\t")

    return(0)

if __name__ == "__main__":
    exit(main())