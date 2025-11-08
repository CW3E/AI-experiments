# Exploring forecast error origins and large-scale dynamics of weather extremes with an AI weather model
This repository contains the scripts that generate the main results presented in the manuscript entitled **Exploring forecast error origins and large-scale dynamics of weather extremes with an AI weather model**. Provided code covers data preparation, GraphCast model inference, physical diagnosis, and visualization. The following are the specific contents of each folder:

* data_preparation: Scripts that download ERA5 data, generate the adjusted climatology for climatology constraint (CC) experiments, and convert them into a structure usable by GraphCast.
* graphcast: A modified rollout.py script to replace the script in the GraphCast source code for constrained experiments.
* inference: Scripts to generate free or constrained predictions using GraphCast.
* diagnosis: Scripts to calculate apparent heat source (Q1).
* figures: Scripts that generate main figures in the manuscript.
* utils: Some auxiliary functions.

This repository is released under the UCSD license, except for certain files derived from DeepMind’s GraphCast project, which are distributed under the Apache License 2.0. Those files retain their original license headers.
