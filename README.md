# Scale-Aware Transformers for Diagnosing Melanocytic Lesions



![](figure/pipeline.pdf)

## Introduction

We introduce a novel self-attention-based network to learn representations from digital whole slide images of melanocytic skin lesions at multiple scales. Our model softly weighs representations from multiple scales, allowing it to discriminate between diagnosis-relevant and -irrelevant information automatically. same cases in an independent study.

## Installation

This repo requires the following packages:

- python 3.7.6
- numpy 1.19.2
- opencv-python 4.2.
- pillow 6.1.0
- pytorch 1.7.1
- CUDA 10.2
- NVIDIA GeForce RTX 2080 Ti

## Preprocessing

This Step is highly dependent on the format of slide types. In this work, we use tissue regions per slide at resolution of x20 and downsize to get x12.5, x10, x7.5 and x5. Otsu threshold is used to [segment individual tissue slices](https://digitalslidearchive.github.io/HistomicsTK/examples/simple_tissue_detection.html) from a slide. 

