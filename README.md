# PA22_replication
Replication package for "Face detection, tracking, and classification from large-scale news archives for analysis of key political figures"

---
contributors:
  - Andreu Girbau
  - Tetsuro Kobayashi
  - Yusuke Matsui
  - Benjamin Renoust
  - Shin'ichi Satoh
---

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TeleStats/PA22_replication/blob/main/PA22_replication.ipynb)

## Overview

The code in this replication package for "Face detection, tracking, and classification from large-scale news archives for analysis of key political figures" using Python.
We provide a google colab notebook with the instructions to reproduce the experiments from tables 3-6 and figures 5-6.

The replicator should be able to replicate the results for the different configurations by specifying **channel**, **detector**, and **classifier** presented in the paper.

## Data Availability and Provenance Statements

This paper contains analysis on copyrighted data of TV News, therefore raw video frames are not provided for the replication package. 
Instead, we provide pickle (.pkl) files containing the information of all detected faces in the video excerpts used to test our system. 

This information is the **faces bounding box** located in time and space, and the **face embeddings**, as described in the paper.

The experiments presented in the paper can be reproduced by using this intermediate representation of the data. 

### Statement about Rights

- [x] I certify that the author(s) of the manuscript have legitimate access to and permission to use the data used in this manuscript.
- [x] I certify that the author(s) of the manuscript have documented permission to redistribute/publish the data contained within this replication package.


### License for Data

The data are licensed under a MIT license. See LICENSE.txt for details.


### Summary of Availability

- [ ] All data **are** publicly available.
- [x] Some data **cannot be made** publicly available.
- [ ] **No data can be made** publicly available.

### Details on each Data Source

| Data.Name          | Data.Files  | Location                                        | Provided | Citation |
|--------------------|-------------|-------------------------------------------------|----------|-----|
| “Video frames”     | .jpg / .png | data/dataset/train/channel/year                 | FALSE    | This work |
| “Frames metadata”  | .metaPA22   | data/dataset/train/channel/year                 | TRUE     | This work    |
| “Annotations”      | .csv        | data/dataset/train/channel/year                 | TRUE     | This work    |
| “Face information” | .pkl        | data/results/channel/train/detector-features    | TRUE     | This work    |
| “Face models JP”   | .pkl        | data/resources/face_models/faces_politicians    | TRUE     | This work    |
| “Face models US”   | .pkl        | data/resources/face_models/faces_us_individuals | TRUE     | This work    |

### Dataset collected by the authors

The annotations for the Japanese TV data (NHK, Hodo Station) used to support the findings of this study is publicly available.

### Dataset provided by Hong et al. (2020)

The annotations for the US data (CNN, FOX, MSNBC) used to support the findings of this study is property of Hong et al. (2020), therefore not provided in the public reproduction package of this work.

## Software requirements

Our code is tested with python3.8, and the libraries required are stated in "requirements.txt". 

We provide a [**google colab notebook**]((https://colab.research.google.com/github/TeleStats/PA22_replication/blob/main/PA22_replication.ipynb)) to reproduce the experiments of this paper, and strongly recommend following it.

## Description of programs/code

Code is included in the `src` folder.

- Code in `face_classifier.py` tracks and classifies the face detections along a video. 
- Code in `metrics.py` runs the evaluation for the specified options (detector + classifier). 
- Code in `convert_results_to_fiftyone.py` converts the datasets to "fiftyone" format in order to reproduce figures 5-6.  
- Code in `convert_dataset_to_fiftyone.py` populates the dataset with the detections/classification of the key individuals for reproducing figures 5-6.
- The rest of files contain classes and functions used by the previously mentioned scripts.

### License for Code

The code is licensed under a MIT license. See [LICENSE.txt](LICENSE.txt) for details.

## Instructions to Replicators

Classification should be run for all the different configurations (detector-feats-classifier), e.g. yolo-resnetv1-fcg_average_vote. 
The step-by-step instructions are defined in the google colab notebook, please follow those in order to reproduce our results.

## List of tables and programs

Each number of tables 3-6 is extracted from the configuration triplet detector-feats-classifier. 
To check a specific number, please run the face classifier and metrics for the configuration triplet to be checked.

The provided code reproduces:

- [X] All numbers provided in text in Section 4.4 (Analysis) in the paper.
- [X] All tables and figures in Section 4.4 (Analysis) in the paper.

## References

Hong, J., W. Crichton, H. Zhang, D. Y. Fu, J. Ritchie, J. Barenholtz, B. Hannel, et al. 2021. “Analysis of Faces
in a Decade of US Cable TV News.” In Proceedings of the 27th ACM SIGKDD International Conference on
Knowledge Discovery Data Mining. Association for Computing Machinery.
