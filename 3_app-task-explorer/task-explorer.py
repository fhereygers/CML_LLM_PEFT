import gradio as gr
import os
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from glob import glob
from collections import namedtuple 


LORA_ADAPTERS_DIR = "./adapters_prebuilt"
if os.path.exists(LORA_ADAPTERS_DIR):
  print("Found adapters")
  
lora_adapters = glob(LORA_ADAPTERS_DIR+"/*/", recursive = False)


model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-1b1", return_dict=True, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-1b1")

for adapter in lora_adapters:
  name = adapter_name=os.path.basename(adapter.strip("/"))
  # See https://github.com/huggingface/peft/issues/211
  # This is a PEFT Model, we can load another adapter
  if hasattr(model, 'load_adapter'):
    model.load_adapter(adapter, adapter_name=name)
  # This is a regular AutoModelForCausalLM, we should use PeftModel.from_pretrained for this first adapter load
  else:
    model = PeftModel.from_pretrained(model=model, model_id=adapter, adapter_name=name)

loaded_adapters = list(model.peft_config.keys())

def generate(prompt, max_new_tokens, temperature, repetition_penalty):
  batch = tokenizer(prompt, return_tensors='pt')
  with torch.cuda.amp.autocast():
    output_tokens = model.generate(**batch, max_new_tokens=max_new_tokens, repetition_penalty=repetition_penalty,temperature=temperature )
  prompt_length = len(prompt)
  return tokenizer.decode(output_tokens[0], skip_special_tokens=True)[prompt_length:]

def get_responses(adapter_select, prompt, max_new_tokens, temperature, repetition_penalty):
  # Using this syntax to ensure inference without adapter
  # https://github.com/huggingface/peft/issues/430
  with model.disable_adapter():
    base_generation = generate(prompt, max_new_tokens, temperature, repetition_penalty)
  
  if "bloom1b1-lora-instruct" in adapter_select:
    model.set_adapter("bloom1b1-lora-instruct")
  elif "bloom1b1-lora-sql" in adapter_select:
    model.set_adapter("bloom1b1-lora-sql")
  elif "bloom1b1-lora-toxic" in adapter_select:
    model.set_adapter("bloom1b1-lora-toxic")

  if "None" in  adapter_select:
    lora_generation = ""
  else:
    lora_generation = generate(prompt, max_new_tokens, temperature, repetition_penalty)
  return (gr.Textbox.update(value=base_generation, visible=True), gr.Textbox.update(value=lora_generation, visible=True))

theme = gr.themes.Default().set(
    block_title_padding='*spacing_md',
)

with gr.Blocks(theme=theme) as demo:
    with gr.Row():
        gr.Markdown("# Fine-tuned LLM Adapters For Multiple Tasks")
    with gr.Column():
        with gr.Box():
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        gr.Markdown("## Use Case")
                    usecase_select = gr.Dropdown(["General Instruction-Following", "Generate SQL given a question and table", "Detoxify Statement"], value="Please select a task to complete...", label="Generative AI Task", interactive=True)
                    input_txt = gr.Textbox(label="Engineered Prompt", value="Select a task example above to edit...", lines=8, interactive=False)
                    with gr.Accordion("Advanced Generation Options", open=False):
                        with gr.Column():
                            with gr.Row():
                                max_new_tokens = gr.Slider(
                                    minimum=0, maximum=256, step=1, value=50,
                                    label="Max New Tokens",
                                )
                                num_beams = gr.Slider(
                                    minimum=1, maximum=10, step=1, value=1,interactive=False,
                                    label="Num Beams (wip)",
                                )
                                repetition_penalty = gr.Slider(
                                    minimum=0.01, maximum=4.5, step=0.01, value=1.1,
                                    label="Repeat Penalty",
                                )

                            with gr.Row():
                                temperature = gr.Slider(
                                    minimum=0.01, maximum=1.99, step=0.01, value=0.7,
                                    label="Temperature",
                                )

                                top_p = gr.Slider(
                                    minimum=0, maximum=1, step=0.01, value=1.0, interactive=False,
                                    label="Top P (wip)",
                                )

                                top_k = gr.Slider(
                                    minimum=0, maximum=200, step=1, value=0, interactive=False,
                                    label="Top K (wip)",
                                )
                    
                with gr.Column():
                    with gr.Row():
                        gr.Markdown("## Inference")
                    with gr.Row():
                        with gr.Column():
                            base_model = gr.TextArea(label="\tBase Model", value="bigscience/bloom1b1", container = False, lines=1, visible=True, interactive=False)
                            output_plain_txt = gr.Textbox(value="", label="Inference", lines=1, visible=False)
                    with gr.Row():
                        with gr.Column():
                            adapter_select = gr.TextArea(label="\tPEFT[LoRA] Adapter", container = False, value="...", lines=1, visible=True, interactive=False)
                            output_adapter_txt = gr.Textbox(value="", label="Inference",lines=1, visible=False)
                    with gr.Row():
                        with gr.Column():
                            clear_btn = gr.ClearButton(value="Reset", components=[], queue=False)
                            gen_btn = gr.Button(value="Generate", variant="primary", interactive=False)



    with gr.Accordion("Documentation", open=False):
        with gr.Row():
            gr.Markdown("# Prompt Examples")
        with gr.Row():
            gr.Markdown("Each of the demo LoRA adapters has been fine-tuned using techniques optimized for cost and time. Even the simple and fast fine-tuning demonstrated in the tutorial code show initial progress in performing bespoke tasks better than the original small foundation model bloom-1b1.")
        with gr.Row():
            gr.Markdown("#### Select an Task Adapter in the dropdown above to load an example and click generate to compare inference on the foundation bloom-1b1 model and the fine-tuned adapters.")

        
        with gr.Row():
            with gr.Column():
                gr.Markdown("## General Instruction-Following (bloom1b1-lora-instruct)")
                gr.Markdown("This demo bloom-1b1_lora_instruct LoRA adapter has been fine-tuned on a fraction of the --- dataset to attempt some intial ability at following instructions.")

            with gr.Column():
                gr.Markdown("## Generate simple SQL (bloom1b1-lora-sql)")
                gr.Markdown("This demo bloom1b1-lora-sql LoRA adapter has been fine-tuned on the on a fraction of the --- dataset to generate valid SQL queries for a question about a given table.")

            with gr.Column():
                gr.Markdown("## Detoxify a passage (bloom1b1-lora-toxic)")
                gr.Markdown("This demo bloom1b1-lora-toxic LoRA adapter has been fine-tuned on a fraction of the --- dataset to detoxify a passage by rephrasing or removing toxic language.")
        
    examples_params_list= [input_txt, repetition_penalty, temperature, max_new_tokens]
    example_tuple = namedtuple("example_named",["input_txt", "repetition_penalty", "temperature", "max_new_tokens", "placeholder_txt"])
    ex_instruct = example_tuple("<Instruction>: Answer the question using the provided input, be concise. How does CML unify self-service data science and data engineering?\n\n<Input>: Cloudera Machine Learning is Cloudera’s cloud-native machine learning platform built for CDP. Cloudera Machine Learning unifies self-service data science and data engineering in a single, portable service as part of an enterprise data cloud for multi-function analytics on data anywhere.\n\n<Response>:", 0.99, 0.75, 33, "")
    ex_sql = example_tuple("<TABLE>: CREATE TABLE jedi (id VARCHAR, lightsaber_color VARCHAR)\n<QUESTION>: Give me a list of jedi that have gold color lightsabers.\n<SQL>: ", 1.15, 0.8, 14, "")
    ex_toxic = example_tuple("<Toxic>: I hate Obi Wan, he always craps on me about the dark side of the force.\n<Neutral>: ", 1.22, 0.75, 19, "")
    ex_empty = example_tuple("",1.0, 0.7, 50,"Select a task example above to edit...")

    def set_example(adapter):
        interactive_prompt = True
        if "bloom1b1-lora-instruct" in adapter:
            update_tuple = ex_instruct
        elif "bloom1b1-lora-sql" in adapter:
            update_tuple = ex_sql
        elif "bloom1b1-lora-toxic" in adapter:
            update_tuple = ex_toxic
        else:
            interactive_prompt = False
            update_tuple = ex_empty

        return (gr.Textbox.update(value=update_tuple.input_txt, interactive=interactive_prompt, placeholder=update_tuple.placeholder_txt),
                gr.Slider.update(value=update_tuple.repetition_penalty),
                gr.Slider.update(value=update_tuple.temperature),
                gr.Slider.update(value=update_tuple.max_new_tokens),
                gr.Textbox.update(value="", visible=False),
                gr.Textbox.update(value="", visible=False))
    import time
    def set_usecase(usecase):
        # Slow user down to highlight changes
        time.sleep(0.5)
        if "General Instruction-Following" in usecase:
            return (gr.Textbox.update(value="bloom1b1-lora-instruct", visible=True), gr.Button.update(interactive=True))
        elif "Generate SQL given a question and table" in usecase:
            return (gr.Textbox.update(value="bloom1b1-lora-sql", visible=True), gr.Button.update(interactive=True))
        elif "Detoxify Statement" in usecase:
            return (gr.Textbox.update(value="bloom1b1-lora-toxic", visible=True), gr.Button.update(interactive=True))
        else:
            return (gr.TextArea.update(value="...", visible=True), gr.Button.update(interactive=False))
    
    def clear_out():
        empty_example = set_example("")
        cleared_tuple = empty_example + (gr.TextArea.update(value="..."), gr.TextArea.update(value="", visible=False), gr.Textbox.update(value="", visible=False), gr.Textbox.update(value="Please select a fine-tuned adapter...")) 
        return cleared_tuple
    
    def show_outputs():
        return (gr.Textbox.update(visible=True), gr.Textbox.update(visible=True))
    
    def disable_gen():
        return gr.Button.update(interactive=False)
    
    usecase_select.change(set_usecase, inputs = [usecase_select], outputs=[adapter_select, gen_btn])

    adapter_select.change(set_example, inputs = [adapter_select], outputs=[input_txt, repetition_penalty, temperature, max_new_tokens, output_plain_txt, output_adapter_txt])

    clear_btn.click(disable_gen, queue = False, inputs = [], outputs=[gen_btn]).then(clear_out, queue = False, inputs = [], outputs=[input_txt, repetition_penalty, temperature, max_new_tokens, output_plain_txt, output_adapter_txt, adapter_select, output_adapter_txt, output_plain_txt, usecase_select])

    gen_btn.click(show_outputs, inputs = [], outputs=[output_plain_txt,output_adapter_txt]).then(get_responses, inputs=[adapter_select, input_txt, max_new_tokens, temperature, repetition_penalty],
                        outputs=[output_plain_txt,output_adapter_txt])

demo.launch(server_port=int(os.getenv('CDSW_APP_PORT')),
           enable_queue=True,
           show_error=True,
           server_name='127.0.0.1',
)