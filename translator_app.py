import gradio as gr
import my_translator

translator = my_translator.Translator("1.3b")

direction = ("", "")
def translate(text: str, source: str, target: str) -> str:
    global direction
    if direction != (source, target):
        direction = (source, target)
        translator.set_direction(source, target)
        print(f"Direction is changed: {direction}")
    return translator.translate(text)

app = gr.Interface(
    fn=translate,
    inputs=[
        "text",
        gr.Dropdown(my_translator.language_list, multiselect=False, label="Source language"),
        gr.Dropdown(my_translator.language_list, multiselect=False, label="Target language",)
    ],
    outputs=["text"]
)

app.launch()
