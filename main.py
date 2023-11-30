import os

import gradio as gra



OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def main():
    from Promt_prototype.manager import run_flow
    # define gradio interface and other parameters
    app = gra.Interface(fn=run_flow, inputs="text", outputs="text")
    app.launch(
        share=True
    )


if __name__ == "__main__":
    main()