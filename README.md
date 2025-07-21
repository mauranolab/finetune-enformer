# Finetune Enformer

Enformer model fine-tuning code to train on experimentally evaluated synthetic constructs delivered to a genomic locus.
We developed a fine-tuning strategy to improve performance by incorporating synthetic regulatory genomics datasets. We added a new independent output layer that uses the baseline Enformer feature extraction trunk to predict our synthetic assays expression data. The new output layer is composed of a self-attention layer to capture relevant features independently of position and a dense layer to combine the resulting signal into a single prediction value. We evaluated three configurations of our new output self-attention layer: **SingleHead 64/64**, **SingleHead 64/128**, and **MultiHead 64/64**. **SingleHead 64/64** applies a single projection of 64 key and value matrices, **SingleHead 64/128** applies a single projection of 64 key and 128 value matrices, and **MultiHead 64/64** applies four independent projections of 64 key and value matrices.

## Installation

### Dependencies

We recommend using `python 3.9` due to several dependencies of Enformer original [code](https://github.com/google-deepmind/deepmind-research/tree/master/enformer).
All other library requirements are listed under `requirements.txt`.

To install them run the following:

```bash
## Create a virtual environment to isolate your packages
python3 -m venv .venv
source .venv/bin/activate
## Install requirements
pip install requirements.txt
## Check installation
python3 -m src.models.train --help
```

### Setup

The code requires two environment variables be defined, to locate Enformer
tensorhub and publication weights. They can be defined as follow:

```bash
## Address to Enformer tensorhub version. When using `tensorhub` model,
## the program will fetch and cache from the specified address.
ENFORMERTENSORHUB="https://www.kaggle.com/models/deepmind/enformer/TensorFlow2/enformer/1"
## Local path to Enformer publication weights from their
## [google storage](gs://dm-enformer/models/enformer/sonnet_weights/)
ENFORMERBASELINE="data/Avsec2021Weights/"
```

Other than those, you may consider define the following variables to simplify
usage:

```bash
SEQ_LENGTH=25600 ## Prediction input/output size in bp
SEQ_WINDOW=128   ## Prediction output bin size. Default produced by Enformer architecture

## Fine-tuned models attention layer specifications.
## When using our code, you may be required to define them whenever running a
## specific model.
## - SingleHead 64/64
SPEC_SH6464="--key-size 64  --value-size 64  --num-heads 1"
## - SingleHead 64/128
SPEC_SH64128="--key-size 64 --value-size 128 --num-heads 1"
## - MultiHead 64/64
SPEC_MH6464="--key-size 64  --value-size 64  --num-heads 4"
```

## Usage

### Building a Fine-Tunning Dataset

First we prepare the environment and define some reference requirements.

```bash
## Define a seed to guarantee  reproducibility
SEED1=4200
SEED2=5200
MM10REF="/path/to/mm10.fasta"
Sox2LCR_COORDINATES="chr3:34732778-34772706"

## Create a dataset directory
mkdir -p data/dataset
```

Next we build a synthetic payload dataset by replacing the endogenous **Sox2 LCR** locus with our synthetic payloads replicating assays from [_Brosh et al. 2023_](https://www.cell.com/molecular-cell/fulltext/S1097-2765(23)00154-5).
By default, the generated sequences are 28,880bp long (25,600 target window +
1280 padding) centered at the replaced locus.

```bash
python3 -m src.data.make_dataset \
    ## Payload activity reference table. A TAB delimited file, requiring 
    ## `MenDel.Name`, `group`, `foldchange`, `activity` fields to be defined.
    data/raw/Brosh2023_TableS4.tsv \
    ## Payloads sequence fasta. Sequence names must correspond to `MenDel.Name`
    ## from activity table.
    data/raw/Brosh2023_Payload.fasta \
    ## BED Annotation of relevant locus from payloads.
    --annotation data/raw/Sox2_DistalLCR.bed \
    ## Reference genome which the payloads will be inserted.
    --reference ${MM10REF} \
    ## Genomic coordinates to be replaced by payloads.
    --context $Sox2LCR_COORDINATES \
    ## Seed to guarantee results reproducibility. It is relevant when assigning
    ## payloads to training or testing folds
    --seed ${SEED1} \
    ## folds assignment ratio. Due to payload diversity constraints we assigned
    ## everyone to a training fold and sampled two set of sequences to training
    ## and testing.
    --folds train=1 \
    --output data/dataset/Brosh2023.tsv
```

From this dataset, we sampled 48 random strides of 25,600bp for each payload to
be a training dataset and other 48 strides for testing.

```bash
## Will generate a sequence (`{prefix}-sequence.{fold}.npy`) and activity
## (`{prefix}-activity.{fold}.npy`) employing during training and testing.
python3 -m src.data.make_dataset_array data/dataset/Brosh2023.tsv \
    --prefix data/dataset/Brosh2023 \
    --seed $SEED1 \
    --stride 1 \
    --length 25600 \
    --sample 48

cat data/dataset/Brosh2023.tsv | sed 's/train/validation/' > tmp.tsv
python3 -m src.data.make_dataset_array tmp.tsv \
    --prefix data/dataset/Brosh2023 \
    --seed $SEED2 \
    --stride 1 \
    --length 25600 \
    --sample 48
```

### Fine-Tunning training

After generating a training and validation dataset, we can fine tune Enformer
weights based on our dataset with the command below:

```bash
MODEL="out/SingleHead64_64" ## Path to model weights
## Fine-tuned attention layer specifications
MODELSPEC="--key-size 64  --value-size 64  --num-heads 1"

python3 -m src.models.train \
    ${MODEL} data/dataset/Brosh2023 ${MODELSPEC} \
    --learning-rate 1E-5 \
    ## Number of fine-tunning epochs
    --epochs 10 \
    ## Number of evaluations per epoch
    --steps 100 \
    ## Number of entries evaluated simultaneously
    --batch 4
```

### Evaluating results

After fine-tuning, we can use the following code to evaluate our model predictions with the original dataset results.
It outputs a table consisting of dataset info (payload, group, fold, and activity) together with evaluated sequence offset, directly predicted signal and sum of maximum values from all annotated sites (here Sox2 LCR DHSs).

```bash
python3 -m src.models.predict \
    ${MODEL} data/dataset/Brosh2023.tsv ${MODELSPEC} \
    ## Enformer output track of interest
    --head mouse --track 10 \
    ## Number of samples and strides to be evaluated per payload
    --sample 32 --stride 1 \
    --seed $SEED1 \
    --output out/SingleHead64_64/Brosh2023.tsv
```

## Credits

This project Enformer implementation is based on their code available
[here](https://github.com/google-deepmind/deepmind-research/tree/master/enformer)
with only a minor modification to remove input/output size constraints (see
[enformer.py](src/models/enformer.py) at line 141).

## Citation

If you find this repository useful and use some of our code, please cite our
[preprint](https://www.biorxiv.org/content/10.1101/2025.02.04.636130v2):

> Ribeiro-dos-Santos, André M., and Matthew T. Maurano. Iterative Improvement
> of Deep Learning Models Using Synthetic Regulatory Genomics.
> bioRxiv, 21 Feb. 2025. bioRxiv, https://doi.org/10.1101/2025.02.04.636130.