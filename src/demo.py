import gradio as gr
from sample import generate_gpt2


app = gr.Blocks()
with app:
    with gr.Row():
        input_text = gr.Textbox(lines=2, label="Input Prompt")
    with gr.Row():
        btn_gen = gr.Button(value="Generate", variant="primary")
    with gr.Row():
        with gr.Column():
            output_text_model1 = gr.Textbox(lines=5, label="Base GPT-2")
        with gr.Column():
            output_text_model2 = gr.Textbox(lines=5, label="SFT GPT-2")

    btn_gen.click(generate_gpt2, inputs=input_text, outputs=[output_text_model1, output_text_model2])
    # gr.Interface(generate_gpt2, inputs=input_text, outputs=[output_text_model1, output_text_model2], 
                # title="Text Generation Comparison", description="Enter a prompt and compare the generated text from two different models.").launch()
app.launch()
