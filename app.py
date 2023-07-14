import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(
    "Salesforce/xgen-7b-8k-base", 
    trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    "Salesforce/xgen-7b-8k-base", 
    torch_dtype=torch.bfloat16)

def summarizer(text):
    header = (
        "A chat between human and AI"
    )

    text = header + "### Human: Please summarize the following article. \n\n" + text + "\n###"  
    inputs = tokenizer(text, return_tensors="pt") 
    generated_id = model.generate(
        **inputs, 
        max_length=1024,
        do_sample=True,
        top_p=0.95,
        top_k=50,
        temperature=0.7)

    summary = tokenizer.decode(generated_id[0],skip_special_tokens=True).lstrip()
    summary = summary.split("### Assistant:")[1]
    summary = summary.split("<|endoftext|>")[0]
    
    return gr.Textbox.update(value=summary)


with gr.Blocks() as final:
    with gr.Row():
        text = gr.Textbox(lines=20,label="Input Text")
        summary = gr.Textbox(label="Summary",lines=20)
    submit = gr.Button(text="Summarize")
    submit.click(summarizer,inputs=text,outputs=summary)

final.launch()

