# A Large Language Model for Feasible and Diverse Population Synthesis

This repository accompanies the paper **“A Large Language Model for Feasible and Diverse Population Synthesis.”**  
It provides code and data to reproduce every experiment that compares five generative approaches for synthetic-population creation.

## Overview
We benchmark the following models:

* **Vanilla-VAE** — a baseline variational autoencoder.  
* **Vanilla-WGAN** — a Wasserstein GAN trained with gradient penalty.  
* **LLM-Random** — a lightweight fine-tuned LLM(GPT2) without structural guidance and using random ordering during autoregressive encoding.  
* **LLM-BN** — a lightweight fine-tuned LLM(GPT2) that imposes a Bayesian-Network topological ordering during autoregressive encoding.  
* **LLM-Few-shot** — prompt-based generation with a handful of curated examples.

The goal is to generate population by balancing between **feasibility** and **diversity**.

## Data
* **Raw survey upload** – The `data/` directory contains an anonymised, pre-processed extract of the *South Korean National Household Travel Survey (HTS)*.  
  *All personally identifiable information was removed before distribution.*
* **`h-population.csv`** – The full reference population created by pooling the 2010, 2016, and 2021 waves of the Korean HTS.
* **`h-sample.csv`** – A 5 % simple-random sample drawn from `h-population.csv`; **every model in this study is trained exclusively on this identical sample**.
* **Generated synthetic data** – After training, each model writes its output to **`generated_synthetic_data_<MODEL_NAME>.csv`**.
* For the `LLM-Few-shot` model, generating a dataset the size of `h-population` in one pass is infeasible, so multiple parallel runs are launched with user-defined chunk sizes, and the resulting files are merged afterward: **`generated_synthetic_data_LLM-Few-shot.csv`**
