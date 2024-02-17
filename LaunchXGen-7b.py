import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(
    "Salesforce/xgen-7b-4k-base", trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    "Salesforce/xgen-7b-4k-base", torch_dtype=torch.bfloat16) #load_in_8bit=True

def talkto(text):
    #header =("Hi Gen!")
    #text=header + " I'am Ri\n\n" + text +"\n"
    inputs = tokenizer(text, return_tensors="pt")
    sample = model.generate(
        **inputs, 
        max_length=1024,
        do_sample=True, 
        top_p=0.95, 
        top_k=40, 
        temperature=0.7
        )
    output=tokenizer.decode(sample[0]) #,skip_special_tokens=True).lstrip()
    return gr.Textbox(value=output)

def test(text):
    output ="testing..."
    return gr.Textbox(value=output)

with gr.Blocks() as demo:
    with gr.Row():
        text=gr.Textbox(lines=15, label="Me")
        output=gr.Textbox(label="Gen",lines=15)
    submit=gr.Button("Send")
    submit.click(talkto,inputs=text, outputs=output)

demo.launch()