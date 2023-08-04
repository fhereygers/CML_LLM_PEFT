# LLM Task Fine-tuning in CML with PEFT

This repository demonstrates how to use [PEFT](https://huggingface.co/blog/peft) (Parameter-Efficient Fine-Tuning) and distribution techniques to fine-tune open source LLM (Large Language Model) for downstream language tasks.
## Overview
### Why Fine-tune a Foundation LLM?
While foundation LLMs are powerful and can generate very convincing language as a result of expensive and extensive training, they are not always suited for the specific downstream tasks that a generative AI application may require.

Fine-tuning to create models suitable for specific tasks is becoming increasingly more accessible, in the [QLoRA (Quantized Low-Rank Adaptation) Paper](https://arxiv.org/abs/2305.14314), researchers were able to finetune a 65B parameter model on a Single 48GB GPU while reaching 99% of the performance level of ChatGPT compared to the 780GB of GPU memory required to perform full 16-bit finetuning techniques. This means a cost difference of over 16x in baremetal GPU alone. 
## AMP Overview
In this AMP we show you how to implement LLM fine-tuning jobs that make use of the QLoRA and Accelerate implementations available in the PEFT open-source library from Huggingface.

The fine-tuning examples for 3 different tasks are created as CML Jobs that can be run to reproduce the sample model adapters included in this AMP repo in [./adapters_prebuilt](./adapters_prebuilt).

## AMP Requirements

### CPU
- CML CPU workloads with resource profiles up to (2 vCPU / 18 GiB Memory) will be provisioned
### GPU
-  Minimum of nVidia V100 with 16GB vram is required (AWS p3.2xlarge)
- 1+ CML GPU workloads with resource profile (2 vCPU / 16 GiB Memory / 1 GPU) will be provisioned
  - Fine-tuning Examples (Optional)
    - A single gpu will run fine-tuning examples only in non-distributed mode
    - Multiple gpus will be required to run fine-tuning examples distributed across multiple CML sessions.
  - Application Inference
    - The task explorer application will require 1 GPU to perform inference
### CML Runtime
Workbench - Python 3.9 - Nvidia GPU - 2023.05
## AMP Setup  
### Configurable Options
**NUM_GPU_WORKERS:** Configurable project environment variable set up for this AMP. This is the number of distributed GPUs that the fine-tuning jobs will make use of during runtime.
## AMP Details
### Fine-tuning optimization techniques
In this AMP we show how you can use cutting edge fine-tuning techniques to effictiently produce adapters finetuned for language tasks in CML.
- #### PEFT
  PEFT (Parameter-Efficient Fine-Tuning) are a class of fine-tuning techniques which allow for the effecient adaptation of LLMs to downstream tasks, training only a small amount of extra model parameters. 

  Full fine-tuning is effective and has been the default way to apply base LLMs to different tasks, but is now seen as  inefficient due to the ballooning size of LLMs and datasets. PEFT techniques offer much more time and cost efficiecnt fine-tuning pipelines and in the case of LoRA, more portable results.

- ##### QLoRA
  One of the PEFT techniques officially supported in the huggingface library is QLoRA (Quantized Low Rank Adaptation). This fine-tuning technique is the result of two papers, the original [LoRA](https://arxiv.org/abs/2106.09685) and following [QLoRA](https://arxiv.org/abs/2305.14314).

  - LoRA fine-tuning freezes the original model parameters and trains a new small set of parameters with an dataset, at lower memory footprint and time therefore lower cost for still very effective leraning.

  - QLoRA further increased efficiency, by using a new quantized data type and additional quanitization and memory optimization techniques to further drive the time and cost down.

  This allows us to use lower cost GPUs compared to full parameter fine-tuning, while still matching the performance of more intensive and costly full fine-tuning.

  All of the libraries required for configuring and launching QLoRA finetuning are available via from huggingface see [requirements.txt](./requirements.txt).

- #### Distributed Training
  Using the PEFT open-source library from Huggingface means we also have easy access to [accelerate](https://github.com/huggingface/accelerate). Another Huggingface library which abstracts away the use of multiple GPUs and machines for fine-tuning jobs. As with many other kinds of distributed workloads, this cuts down on the time to fine-tune dramatically.

  CML can is able to run accelerate distributed fine-tuning workloads out of the box using the [CML Workers API](https://docs.cloudera.com/machine-learning/cloud/distributed-computing/topics/ml-workers-api.html)

## Sample Fine-tuned Tasks
For each the following fine-tuning tasks we start with the *smaller* LLM [bigscience/bloom-1b1](https://huggingface.co/bigscience/bloom-1b1).
This model was chosen for its tiny size and permissive license for. The small size of this base model results in very short fine-tuning times and portable adapters that are simple to run for anyone looking to try this AMP.

A larger base model or a base model from another LLM family could also be used with the same techniques shown in the scripts and sample notebook in this repository.
> An update to make this easier to do within this AMP is coming soon!

Each included sample adapter is fine-tuned on portions of publicly available datasets that have been mapped to fit desired inference patterns. While none of trained adapters are production-level models, each are proof of task performance improvement (* see [Improving on the Sample Adapters](#improving-on-the-sample-adapters)) over the base model even with minimal training time on the scale of minutes.

### General Instruction Following
- Training Time/Cost: (8m19s / $0.82) distributed on 2x P3.2xlarge AWS instances
- Dataset: https://huggingface.co/datasets/teknium/GPTeacher-General-Instruct (mit)
  - Contains 89k examples of instruction-input-response text
### SQL English to Query
- Training Time/Cost: (22m11s / $2.21) distributed on 2x P3.2xlarge AWS instances
- Dataset: https://huggingface.co/datasets/philschmid/sql-create-context-copy (cc-by-4.0)
  - Contains 80k examples of Question - Table - SQL text-to-sql strings
### Detoxifying English Text
- Training Time/Cost: (8m50s / $0.86) distributed on 2x P3.2xlarge AWS instances
- Dataset: https://huggingface.co/datasets/s-nlp/paradetox (afl-3.0)
  - Contains 19k examples of toxic to neutral wording conversions in english

## Implementation
See detailed implementation descriptions in [distributed_peft_scripts/README.md](./distributed_peft_scripts/README.md)

## Jupyter Notebook Example

A [notebook example](fine_tune_sample.ipynb) is provided to demonstrate what the fine-tuning techniques and libraries look like in a single script.
### Recommended Runtime
Jupyter Lab - Python 3.9 - Nvidia GPU - 2023.05
### Recommended Resource Profile
2 vCPU / 16 GiB Memory / 1 GPU