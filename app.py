import os
import gradio as gr
import subprocess
os.system("git clone https://github.com/doevent/FullSubNet-plus")
os.system("mv FullSubNet-plus/speech_enhance .")
os.system("mv FullSubNet-plus/config .")
os.system("gdown https://drive.google.com/uc?id=1UJSt1G0P_aXry-u79LLU_l9tCnNa2u7C -O best_model.tar")
from speech_enhance.tools.denoise_hf_clone_voice import start


# If the file is too duration to inference
def duration(input_audio) -> int:
    command = f"ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 -i {input_audio}"
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE)
    data = result.stdout.decode('ascii').rstrip()
    return int(float(data))
    

def inference(audio):
    try:
        if audio.find("audio") < 0:
            if duration(audio) >= 150:
                return "error.wav"
        result = start(to_list_files=[audio])
        return result[0]
    except Exception as e:
        return "error.wav"

        
title = """<h1 id="title">DeNoise Speech Enhancement</h1>"""
description = """
This is an unofficial demo for FullSubNet-plus: DeNoise Speech Enhancement. To use it, simply upload your audio, or click one of the examples to load them. Read more at the links below.
Link to GitHub:
- [FullSubNet +](https://github.com/hit-thusz-RookieCJ/FullSubNet-plus)  
"""
twitter_link = "[![](https://img.shields.io/twitter/follow/DoEvent?label=@DoEvent&style=social)](https://twitter.com/DoEvent)"
css = '''
h1#title {
  text-align: center;
}
'''

demo = gr.Blocks(css=css)
with demo:
    gr.Markdown(title)
    gr.Markdown(description)
    gr.Markdown(twitter_link)
    
    
    with gr.Tabs():
        with gr.TabItem("Upload audio"):
            iface = gr.Interface(
                inference,
                inputs = gr.Audio(source="upload", type="filepath"),
                outputs = gr.Audio(type="file"),
                examples=[["man.wav"], ["woman.wav"]])
        with gr.TabItem("Record your voice"):
            iface = gr.Interface(
              inference,
              inputs = gr.Audio(source="microphone", label="Record yourself reading something out loud", type="filepath"),
              outputs = gr.Audio(type="file"),
            )
    
    gr.Markdown("<div><center><img src='https://visitor-badge.glitch.me/badge?page_id=max_skobeev_fullsubnet_plus_public' alt='visitor badge'></center></div>")


demo.launch(enable_queue=True, debug=True, show_error=True)