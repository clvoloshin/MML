# Minimax Model Learning (MML)

## Introduction

This repo is an example/reference implementation of MML based on the Minimax Model Learning paper (https://arxiv.org/abs/2103.02084).

This reference for MML uses the Kernel-based implementation from Proposition E.3 to solve the internal maximization exactly, in expectation. For your own usage, you may need to modify the kernel and NN itself to achieve the best results. You may also consider using competitive gradient descent to approximately solve the minimax problem rather than a kernel-based approach.

## Getting started

### Installation

Tested on python3.8+.
```
python3 -m venv reference-env
source reference-env/bin/activate
pip3 install -r requirements.txt
```

### Examples

```
python mml.py
python mle.py
```


