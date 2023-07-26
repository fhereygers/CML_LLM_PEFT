# Workspace Reqs
- CML CPU Workers: 	m4.2xlarge
  - Really any CPU type is suitable
- CML GPU Workers:	p3.8xlarge
  - This has 4 gpus per instance, we will use 3 for finetuning so this will save time on worker spinup

# Project Reqs
- Clone this project by cloning from Git
- Make sure to enable only the PBJ Python 3.9 GPU Runtime

# Stage 0: Install Prereqs
- Launch a Session with at least 2 CPU, 4 MEM (No GPU needed)
- Run file 0_install-requirements.py

# Stage 1: Finetuning Example (Skippable: see Stage 2 LORA_ADAPTERS_DIR)

- Launch a Session with 2 CPU, 4 MEM, 1 GPU
- Run file 1_launch_all_fine_tuning.py
  - Start with small Bloom foundation model
  - Perform LoRA fine-tuning using 3 different datasets targetted for different kinds of tasks
  - Finetuning runs distributed accross 3 GPUs for time optimization
  - Output is 3 dirs inside ./adapters/ each containining a DIFFERENT LoRA Adapter
- During and after Fine-tuning you can view metrics via tensorboard dashboards via the Session App link
  - To see learning rates, time to completion, etc
- Session can be closed after completion if desired
  
Datasets used:
https://huggingface.co/datasets/qwedsacf/grade-school-math-instructions
https://huggingface.co/datasets/teknium/GPTeacher-General-Instruct
https://huggingface.co/datasets/s-nlp/paradetox

# Stage 2: Adapter Playground App 
- Launch a Session with 2 CPU, 4 MEM, 1 GPU
- Run file 2_app.py
- Uses cached adapters trained ahead of time
  - change LORA_ADAPTERS_DIR to use newly fine-tuned adapters from Stage 1


## Example
Given a prompt with special  token like an assistant chat

`<Instruction>: Use the provided text to answer the question. Does CML enable self-service data science?
<Input>: Cloudera Machine Learning is Cloudera’s cloud-native machine learning platform built for CDP. Cloudera Machine Learning unifies self-service data science and data engineering in a single, portable service as part of an enterprise data cloud for multi-function analytics on data anywhere.
<Response>: `

The foundation model produces random text like

`The response will be sent back to the user via email or SMS.  If you are not sure what your response should look like, please contact us at [email protected]`

But we would like to see generated text like

`CML enables selfservice data science by leveraging its powerful AI-powered machine learning capabilities that seamlessly integrate with existing data pipelines and tools. This allows users to leverage their own expertise and experience to create customized solutions tailored to specific needs.`
> NOTE: The above is not perfect and slightly hallucinatory, but on the right track to usable output for this task, tuned on a tiny subet of a dataset.

