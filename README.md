# Stack Overflow Answers Assessment

This repository contains code for a work submitted as a Master's thesis in ITMO University. 
In this work, we solve the problem of assessing Stack Overflow (SO) answers' quality, formulating it as a ranking task.

We work with the [data](https://github.com/daniel-hasan/dalip-wiki-qa-dataset) presented in a 2017 paper "[A general multiview framework for assessing the quality of collaboratively created content on web 2.0](https://asistdl.onlinelibrary.wiley.com/doi/abs/10.1002/asi.23650)".
We train a line of ranking models based on LMs pretrained on SO data using two ranking approaches — point-wise and pair-wise — and two encoder architectures — bi-encoder and cross-encoder.
We compare our models to a set of simpler baselines.
We also conduct an exploratory analysis of the dataset.

## Repository Structure

The contents of the repository are the following:
* `bibliography` contains a notebook for bibliography export
* `config` contains a config file with variables relevant for the project
* `experiments` contains code for training models and analyzing data
* `results_analysis` contains code for the analysis of the results
* `src` contains reusable code for different stages of the training pipeline.
