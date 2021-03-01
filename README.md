# Overview

This is a project that attempts to reproduce and verify the main claims of ["Dense Passage Retrieval for Open-Domain Question Answering"](https://arxiv.org/pdf/2004.04906.pdf) (Karpukhin et al., EMNLP 2020) with the final report following the format of submissions to the [ML Reproducibility Challenge](https://paperswithcode.com/rc2020). This project was done as a final project for [CSE 517 wi21](https://docs.google.com/document/d/1gBz2w79DBrGjNGq2TMqJBDIWzUGsQacWFAszZKz6OKI/edit), taught by Prof. Noah Smith.

# Installation and Setup

These are install instructions to get *literally everything* working on a fresh VM. The VM we used ran Debian GNU/Linux 10 (buster) as its OS, and in particular, we used an A2 machine with 4 Tesla A100s on Google Cloud Platform to reproduce selected results of the paper. As a warning, this paper's results (and subsequently the scripts included in this repository) *are expensive to run and reproduce*; we used roughly $___ worth of Google Cloud Platform credits to reproduce *selected, not all*, results in the original paper.

### Install git

1. `sudo apt install git-all`
	* Answer "yes" if you're prompted.

2. `source .bashrc`
	* This reloads your shell, and the git command should now work. We use this multiple times throughout the setup process.

### Experiment code

This should be cloned in the home directory of your VM.

1. `git clone https://github.com/peminguyen/CSE517-final-project.git`
	* We need environment.yml for setting up conda.

TODO: Get the BERT base uncased file downloaded

### Setting up conda

Starting from your home directory,

1. `wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh`

2. `bash Miniconda3-latest-Linux-x86_64.sh`
	* Go through the prompts. Leaving everything as defaults should be fine. You should run conda init.

3. `source .bashrc`

4. `conda env create -f CSE517-final-project/environment.yml`
	* There's probably quite a few extraneous packages in this `environment.yml` file, but this is guaranteed to work. Feel free to prune stuff that you find isn't needed, though it'll probably be really time-consuming.

5. `conda activate DPR`

### Getting the data

#### Preprocessed Natural Questions (NQ) data

TODO

#### EfficientQA subset of Wikipedia

TODO

# Running the Experiments

TODO

https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html

