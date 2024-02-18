import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('stabilityai/stablelm-zephyr-3b')
model = AutoModelForCausalLM.from_pretrained(
    'stabilityai/stablelm-zephyr-3b',
    trust_remote_code=True,
    device_map="auto"
)

def test():
    prompt = [{'role': 'user', 'content': 'List 3 synonyms for the word "tiny"'}]
    inputs = tokenizer.apply_chat_template(
        prompt,
        add_generation_prompt=True,
        return_tensors='pt'
    )

    tokens = model.generate(
        inputs.to(model.device),
        max_new_tokens=1024,
        temperature=0.8,
        do_sample=True
    )
    print(tokenizer.decode(tokens[0], skip_special_tokens=False))

def talkto(messages):
    prompt = [{'role': 'Ri', 'content': messages}]
    inputs = tokenizer.apply_chat_template(
        prompt,
        add_generation_prompt=True,
        return_tensors='pt'
    )
    tokens = model.generate(
        inputs.to(model.device),
        max_new_tokens=1024,
        temperature=0.8,
        do_sample=True
    )
    output= tokenizer.decode(tokens[0], skip_special_tokens=False)
    return gr.Textbox(value=output)

with gr.Blocks() as demo:
    with gr.Row():
        message=gr.Textbox(lines=15, label="Me")
        output=gr.Textbox(label="Gen",lines=15)
    submit=gr.Button("Send")
    submit.click(talkto,inputs=message, outputs=output)

demo.launch()