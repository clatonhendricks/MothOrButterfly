__all__ = ['learn', 'classify_image', 'categories', 'image', 'label', 'examples', 'intf']

# Cell
from fastai.vision.all import *
import gradio as gr

# Cell
title = 'Is it a Butterfly or Moth'
desc = '<p style="text-align: center;">Prediction model built using FastAI to predict if its a Butterfly or Moth. (other images will show wrong results, no promises <img src="https://html-online.com/editor/tiny4_9_11/plugins/emoticons/img/smiley-laughing.gif" alt="laughing" />)&nbsp;</p>'

# Cell
learn = load_learner('export.pkl')

# Cell
categories = learn.dls.vocab

def classify_image(img):
    pred,idx,probs = learn.predict(img)
    return dict(zip(categories, map(float,probs)))

# Cell
image = gr.inputs.Image(shape=(192, 192))
label = gr.outputs.Label()
examples = ['Butterfly.jpg','Moth.jpg']

# Cell
intf = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples,title=title,description=desc)
intf.launch()