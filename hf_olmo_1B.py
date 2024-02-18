import hf_olmo
import gradio as gr

from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("allenai/OLMo-1B")
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1B")


def sendmessage(message):
    #inputs = tokenizer(message, return_tensors='pt', return_token_type_ids=False)
    ##optional verifying cuda
    #inputs = {k: v.to('cuda') for k,v in inputs.items()}
    #olmo = olmo.to('cuda')
    #response = model.generate(**inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)
    #print(tokenizer.batch_decode(response, skip_special_tokens=True)[0])
    from transformers import pipeline
    olmo_pipe = pipeline("text-generation", model="allenai/OLMo-1B")
    response =olmo_pipe(message)
    return response
message = "The world is crazy. "
print(sendmessage(message))

def talkto(message):
    #header =("Hi Gen!")
    #message=header + " I'am Ri\n\n" + message +"\n"
    inputs = tokenizer(message, return_tensors="pt",return_token_type_ids=False)
    sample = model.generate(
        **inputs, 
        max_new_tokens=2048,
        do_sample=True, 
        top_p=0.95, 
        top_k=50)
        #temperature=0.7
        
    output=tokenizer.decode(sample[0]) #,skip_special_tokens=True).lstrip()
    return gr.Textbox(value=output)

def test(text):
    output ="testing..."
    return gr.Textbox(value=output)

with gr.Blocks() as demo:
    with gr.Row():
        message=gr.Textbox(lines=15, label="Me")
        output=gr.Textbox(label="Gen",lines=15)
    submit=gr.Button("Send")
    submit.click(talkto,inputs=message, outputs=output)

demo.launch()