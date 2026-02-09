# LFM2-350M
This repository implements single-batch inference engine for [LFM2-350M](https://arxiv.org/abs/2511.23404) in C. This was conceived as a result of the [Batched Inference Engine](https://github.com/marvinmboya/LFMs-Continuous-Batching) (build from first principles in PyTorch) achieving single-batch ~50 Tokens/Second on M2-Pro CPU but only ~4 Tokens/Second on Intel i5-8350U (8). This C implementation also focused on building from first principles with CPU-based inference targeted (for now) and only optimized GEMM APIs invoked. The engine thus achieved ~24 Tokens/Second on M2-Pro CPU and ~13 Tokens/Second on Intel i5-8350U. This provides 3x speedups for Intel without profiling and aggressive optimizations!<br/>

All components built from scratch using pure C: 
- Byte-level BPE tokenizer.
- Hybrid-architecture Dense Model. 
- Postprocessing (prefill/decode).
- Only greedy decoding adopted (no temperature scaling or Top-K sampling).
- Hybrid caching.

## Contents

1. [Getting Started](#1-getting-started)  
&nbsp;&nbsp;&nbsp;&nbsp;1.1 [Setup Instructions](#11-setup-instructions)  
&nbsp;&nbsp;&nbsp;&nbsp;1.2 [Usage](#12-usage) 
2. [Acknowledgements](#2-acknowledgements)  
3. [Conclusion](#3-conclusion)  
## 1. Getting Started

### 1.1 Setup Instructions

### Files Setup 
Download generated FP32 weights and Tokenizer .bin files as a zipped file from Google Drive link:<br/>
[Weights & Tokenizer File.](https://drive.google.com/file/d/1EF5ES19B1ch_PS2B_kjup8HR-DYAzikf/view?usp=sharing)

Alternatively:<br/> 
- Download official weights and tokenizer from LFM2-350M Hugging Face's 🤗 [repository](https://huggingface.co/LiquidAI/LFM2-350M). 
- Git clone Batched Inference Engine repo for LFM2-350M (built from scratch in Pytorch). 
- Git pull ``store-c-bin-extras`` branch then run:
```sh
python model_tok_to_bin.py
```
to generate FP32 weights and Tokenizer .bin files. Finally, move files dir to the root dir for this repo.
 ### Tools SetUp
OpenMP (multithreading package) is automatically setup for gcc upgraded to v15. However, clang needs separate OpenMP setup and isn't bundled by default.<br/>
Brew installs [OpenMP](https://formulae.brew.sh/formula/libomp) via
```sh
brew install libomp
```
Linux (Manjaro) installs [OpenMP](https://archlinux.org/packages/extra/x86_64/openmp/) via
```sh
yay -i openmp
```
The next setup installs highly optimized implementations of the CBLAS (C Language interface for the Basic Linear Algebra Subprograms) GEMM (General Matrix Multiply) APIs. ArmPL Library housing these implementations is designed for Arm-based architectures, and OneMKL designed for Intel-based architectures. The libraries can be installed from their respective websites:<br/>
> [ArmPL Download Page.](https://developer.arm.com/Tools%20and%20Software/Arm%20Performance%20Libraries#Downloads)<br/>
[OneMKL Download Page.](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html)<br/>

Alternatively:<br/>
Using brew (Arm processors):
```sh
brew install --cask arm-performance-libraries
```
Using yay for Linux x86_64 (Manjaro):
```sh 
yay -i extra/intel-oneapi-mkl
```
**NOTE:** Ensure the directories to CBLAS GEMM libraries are correctly pointed to as in the Makefile.
### 1.2 Usage
Open terminal, then run:
```bash
make
```
which runs inference with default prompt ``What is hello in Spanish?``<br/>
To pass prompt, run
```bash
make PROMPT="What is the best landmark in Paris?"
```

## 2. Acknowledgements
First thanks to LiquidAI open-sourcing their foundational models, hence the successful implementation of the [Batched Inference Engine](https://github.com/marvinmboya/LFMs-Continuous-Batching) in PyTorch and this single-batch Inference Engine in pure C. Second thanks to Kay Lack for the C video ["Just enough C to have fun"](https://www.youtube.com/watch?v=5aZiRjgSGQU), an awesome beginner to learning C. Third thanks to Jacob Sorber whose [C videos](https://www.youtube.com/@JacobSorber) were fundamental to reminding myself of C tooling and capabilities. Fourth thanks to Yale's notes on [Data Structures and Algorithms with C](https://cs.yale.edu/homes/aspnes/classes/223/notes.html), a solid reference during the engine implementation.

## 3. Conclusion
Both the [other engine](https://github.com/marvinmboya/LFMs-Continuous-Batching) and this were built from first principles, hence full focus on correctness and building from foundational papers with little focus on optimizations. Given this, inference was CPU-based with no optimizations, i.e. regional compilation or flashattention, used except those essential to keep compute effectively finite (caching). I welcome contributions for both engines as I built them to learn and document such and hence solid contributions will gladly be merged (no slop!). Regards!
