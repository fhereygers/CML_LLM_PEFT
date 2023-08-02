# LLM Task Fine-tuning in CML with PEFT

This repository demonstrates how an open source LLM (Large Language Model) can be efficiently fine-tuned for a variety of tasks using PEFT and distribution techniques on curated datasets.

## Why Fine-tune a Foundation LLM?
While foundation LLMs are powerful and can generate very convincing language as a result of expensive and extensive training, they are not always suited for the specific downstream tasks that a generative AI application may require.

In a previous [CML AMP (LLM_Chatbot_Augmented_with_Enterprise_Data)](https://github.com/cloudera/CML_AMP_LLM_Chatbot_Augmented_with_Enterprise_Data) our document chat bot required the use of an open source LLM that was already fine-tuned for "instruction following" tasks. This instruction-following LLM was the product of fine-tuning an LLM on a dataset that included a variety of examples of differenct instruction interactions. The application architecture we implemented in the AMP supplied facts and context to the fine-tuned LLM via prompting, and the fine-tuned LLM generated text in a pattern that appropriately fit the instructions provided.

\< Make point about bespoke tasks one might want to fine-tune for \>

## Fine-tuning optimization techniques
In this AMP we show how you can use cutting edge fine-tuning techniques to effictiently produce adapters finetuned for language tasks in CML.
### PEFT
PEFT (Parameter-Efficient Fine-Tuning) are a class of fine-tuning techniques which allow for the effecient adaptation of LLMs to downstream tasks, training only a small amount of extra model parameters. 

Full fine-tuning is effective and has been the default way to apply different learnings and tasks to language models, but is now seen as grossly inefficient due to the ballooning size of LLMs. PEFT techniques offer much more efficiecnt processes and more portable results by allowing for two major improvements:
- **Computation constraints:** Training only small modules sized at a fraction of the original model's weights with more efficient representations. Meaning less video memory and time required on expensive GPUs.
- **Portability:** Most PEFT techniques result in 

There are a varierty of PEFT methodologies being researched today and many are being implemented in the [huggingface PEFT library](https://github.com/huggingface/peft) for easy access.

#### QLoRA
One of the PEFT techniques available in the huggingface library is QLoRA (Quantized Low Rank Adaptation). This fine-tuning technique is the result of two papers, the original [LoRA](https://arxiv.org/abs/2106.09685) and following [QLoRA](https://arxiv.org/abs/2305.14314).

- LoRA fine-tuning freezes the original model parameters and trains a new small set of parameters with an dataset, at lower memory footprint and time therefore lower cost for still very effective leraning.

- QLoRA further increased efficiency, by using a new quantized data type and additional quanitization and memory optimization techniques to further drive the time and cost down.

This allows us to use lower cost GPUs at a fraction of the time compared to full parameter fine-tuning, while still matching the performance of more intensive and costly full fine-tuning.

### Distributed Training
Using the PEFT open-source library from Huggingface means we also have easy access to [accelerate](https://github.com/huggingface/accelerate). Another Huggingface library which abstracts away the use of multiple GPUs and machines for fine-tuning jobs. As with many other kinds of distributed workloads, this cuts down on the time to fine-tune dramatically.

# AMP
## Overview
In this AMP we show you how to implement LLM fine-tuning jobs that make use of the QLoRA and Accelerate implementations available in the PEFT open-source library from Huggingface.

The fine-tuning examples for 3 different tasks are created as CML Jobs that can be run to reproduce the sample model adapters included in this AMP repo in [./adapters_prebuilt](./adapters_prebuilt).
## Target Fine-tuned Tasks*
For each the following fine-tuning tasks we start with the *smaller* LLM bigscience/bloom-1b1 which comes in at 2.13 GB. Each adapter is fine-tuned on a publicly available dataset that has been mapped to fit the input and output patterns desired.
### General Instruction Following
### SQL English to Query
### Detoxifying English Text

describe 3 tasks we finetune for
mention the speed and scale
reiterate these tasks represent bespoke tasks that may be specific in pattern or usecase for a user/customer
## Open Source Models and Datasets:
### Base LLM Model
- https://huggingface.co/bigscience/bloom-1b1

### Datasets
- https://huggingface.co/datasets/teknium/GPTeacher-General-Instruct
  - Contains 89k examples of instruction-input-response text
- https://huggingface.co/datasets/s-nlp/paradetox
  - Contains 19k examples of toxic to neutral language conversions
- https://huggingface.co/datasets/philschmid/sql-create-context-copy
  - Contains 78k examples of natural language and sql query translations

# Workspace Requirements
- CML CPU Workers: 	1+ m4.2xlarge or larger recommended
  - This AMP is not CPU intensive so workspaces with smaller CPU instance types 
- CML GPU Workers:	1+ p3.4xlarge or larger required
  - nVidia V100 is required for this AMP
  - A single gpu will run fine-tuning examples only in non-distributed mode
  - Multiple gpus will be required to run fine-tuning examples distributed accross multiple CML: sessopms.

## Setup 
### Runtime
 < suggested runtime and etc >
### GPU distribution
**NUM_GPU_WORKERS:** TConfigurable project environment variable set up for this AMP. This is the number of distributed GPUs that the fine-tuning jobs will make use of during runtime.

## Implementation Details
<<< Link to README.md under distributed_peft_scripts >>>

## Notebook Example
A notebook example is provided to demonstrate what the fine-tuning techniques and libraries look like in a single script. [Notebook Link](fine_tune_sample.ipynb)