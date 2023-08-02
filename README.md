# LLM Task Fine-tuning in CML with PEFT

This repository demonstrates how to use PEFT (Parameter-Efficient Fine-Tuning) and distribution techniques to fine-tune open source LLM (Large Language Model) for downstream language tasks.

## Why Fine-tune a Foundation LLM?
While foundation LLMs are powerful and can generate very convincing language as a result of expensive and extensive training, they are not always suited for the specific downstream tasks that a generative AI application may require.

In a previous [CML AMP (LLM_Chatbot_Augmented_with_Enterprise_Data)](https://github.com/cloudera/CML_AMP_LLM_Chatbot_Augmented_with_Enterprise_Data) our document chat bot required the use of an open source LLM that was already fine-tuned for "instruction following" tasks. This instruction-following LLM was the product of fine-tuning an LLM on a dataset that included a variety of examples of differenct instruction interactions. The application architecture we implemented in the AMP supplied facts and context to the fine-tuned LLM via prompting, and the fine-tuned LLM generated text in a pattern that appropriately fit the instructions provided.

Fine-tuning to create models suitable for specific tasks is becoming increasingly more accessible, in the [QLoRA (Quantized Low-Rank Adaptation) Paper](https://arxiv.org/abs/2305.14314), researchers were able to finetune a 65B parameter model on a Single 48GB GPU while reaching 99% of the performance level of ChatGPT compared to the 780GB of GPU memory required to perform full 16-bit finetuning techniques. This means a cost difference of over 16x in baremetal GPU alone. 

## Fine-tuning optimization techniques
In this AMP we show how you can use cutting edge fine-tuning techniques to effictiently produce adapters finetuned for language tasks in CML.
### PEFT
PEFT (Parameter-Efficient Fine-Tuning) are a class of fine-tuning techniques which allow for the effecient adaptation of LLMs to downstream tasks, training only a small amount of extra model parameters. 

Full fine-tuning is effective and has been the default way to apply different learnings and tasks to language models, but is now seen as grossly inefficient due to the ballooning size of LLMs. PEFT techniques offer much more efficiecnt processes and more portable results by allowing for two major improvements:
- **Computation constraints:** Training only small modules sized at a fraction of the original model's weights with more efficient representations. Meaning less video memory and time required on expensive GPUs.
- **Portability:** Most PEFT techniques result in 

There are a varierty of PEFT methodologies being researched today and many are being implemented in the [huggingface PEFT library](https://github.com/huggingface/peft) for easy access. This library as used in the code in this AMP makes it easy to implement QLoRA and Accelerator distribution fine-tuning with the [SFTTrainer](https://huggingface.co/docs/trl/main/en/sft_trainer) class.

#### QLoRA
One of the PEFT techniques available in the huggingface library is QLoRA (Quantized Low Rank Adaptation). This fine-tuning technique is the result of two papers, the original [LoRA](https://arxiv.org/abs/2106.09685) and following [QLoRA](https://arxiv.org/abs/2305.14314).

- LoRA fine-tuning freezes the original model parameters and trains a new small set of parameters with an dataset, at lower memory footprint and time therefore lower cost for still very effective leraning.

- QLoRA further increased efficiency, by using a new quantized data type and additional quanitization and memory optimization techniques to further drive the time and cost down.

This allows us to use lower cost GPUs compared to full parameter fine-tuning, while still matching the performance of more intensive and costly full fine-tuning.

### Distributed Training
Using the PEFT open-source library from Huggingface means we also have easy access to [accelerate](https://github.com/huggingface/accelerate). Another Huggingface library which abstracts away the use of multiple GPUs and machines for fine-tuning jobs. As with many other kinds of distributed workloads, this cuts down on the time to fine-tune dramatically.

# AMP
## Overview
In this AMP we show you how to implement LLM fine-tuning jobs that make use of the QLoRA and Accelerate implementations available in the PEFT open-source library from Huggingface.

The fine-tuning examples for 3 different tasks are created as CML Jobs that can be run to reproduce the sample model adapters included in this AMP repo in [./adapters_prebuilt](./adapters_prebuilt).
## Target Fine-tuned Tasks
For each the following fine-tuning tasks we start with the *smaller* LLM bigscience/bloom-1b1 which comes in at 2.13 GB. Each adapter is fine-tuned on a publicly available dataset that has been mapped to fit the input and output patterns desired. While none of trained adapters are production-level models, each are able to demonstrate varying improvement (* see Improving on the Sample Adapters) over the base model with training time on the scale of minutes.

### General Instruction Following
- Training Time/Cost: (8m19s / $0.82)
- Dataset: https://huggingface.co/datasets/teknium/GPTeacher-General-Instruct (mit)
  - Contains 89k examples of instruction-input-response text
### SQL English to Query
- Training Time/Cost: (22m11s / $2.21)
- Dataset: https://huggingface.co/datasets/philschmid/sql-create-context-copy (cc-by-4.0)
  - Contains 80k examples of Question - Table - SQL text-to-sql strings
### Detoxifying English Text
- Training Time/Cost: (8m50s / $0.86)
- Dataset: https://huggingface.co/datasets/s-nlp/paradetox (afl-3.0)
  - Contains 19k examples of toxic to neutral wording conversions in english

## Improving on the Sample Adapters
We wanted to make sure to produce examples that could be easily and very cheaply replicated by CML users, so we've made a number of conscious choixes to minimize the time spent on expensive hardware. The following items could be changed to improve performance of the resulting adapters (with some increase in fine-tuning time and hardware cost).
### Different or Large Base LLM
There are many families of opensource base models that are available today (bloom, falcon, llama2 to name a few)
- Each of these model families have released base models of varying size, allowing for better downstream performance for fine-tuned tasks when using the larger variant.
- Larger base models means longer training times and larger GPU memory requirements

### Larger and Better Curated Datasets
Fine-tuning is a wasted effort without a good dataset, larger and better curated datasets are the most critical part of producing continually improving fine-tuning results
- Larger Datasets means longer training times

### Fine-tuning Arguments
We make use of the huggingface fine-tuning libraries in TRL (https://huggingface.co/docs/trl/) and PEFT (https://huggingface.co/docs/peft/index). These libraries help to launch fine-tuning operations and also allow for the modification of training arguments used by the underlying pytorch.

#### - trl
This library implements further conveniences for fine-tuning 
  - [TrainingArguments](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments)
    - Adjusting learning_rate, num_epochs and etc can change the fine-tuning time and resulting adapater performance
  - [SFTTrainer](https://huggingface.co/docs/trl/main/en/sft_trainer)
    - Bonus optimizations like packing are implemented in this library and speed up fine-tuning at the cost of some performance

#### peft
  - [LoRA configuration](https://huggingface.co/docs/peft/conceptual_guides/lora#common-lora-parameters-in-peft)
    - Configurations here allow for customization of how LoRA is applied to the loaded base model
  - [BitsAndBytes configuration](https://huggingface.co/docs/transformers/main_classes/quantization)
    - The Q in QLoRA requires quantization which is controlled by a bitsandbytes configuration
## Open Source Models and Datasets referenced:
### Base LLM Model
- https://huggingface.co/bigscience/bloom-1b1

# Workspace Requirements
- CML CPU Workers: 	1+ m4.2xlarge or larger recommended
  - This AMP is not CPU intensive so workspaces with smaller CPU instance types 
- CML GPU Workers:	1+ p3.4xlarge or larger required
  - nVidia V100 is required for this AMP
  - A single gpu will run fine-tuning examples only in non-distributed mode
  - Multiple gpus will be required to run fine-tuning examples distributed accross multiple CML: sessopms.

## Setup 
### Runtime
Workbench Python 3.9 2023.05 and newer are recommended 
### Configurable Options
**NUM_GPU_WORKERS:** Configurable project environment variable set up for this AMP. This is the number of distributed GPUs that the fine-tuning jobs will make use of during runtime.

## Implementation Details
<<< Link to README.md under distributed_peft_scripts >>>

## Notebook Example
A notebook example is provided to demonstrate what the fine-tuning techniques and libraries look like in a single script. [Notebook Link](fine_tune_sample.ipynb)