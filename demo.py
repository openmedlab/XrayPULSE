import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr

from xraypulse.common.config import Config
from xraypulse.common.dist_utils import get_rank
from xraypulse.common.registry import registry
from xraypulse.conversation.conversation import Chat, CONV_ZH

# imports modules for registration
from xraypulse.datasets.builders import *
from xraypulse.models import *
from xraypulse.processors import *
from xraypulse.runners import *
from xraypulse.tasks import *


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


# ========================================
#             Model Initialization
# ========================================

print('Initializing Chat')
args = parse_args()
cfg = Config(args)

model_config = cfg.model_cfg
print(model_config)
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
print(model_cls)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

vis_processor_cfg = cfg.datasets_cfg.openi.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
print('Initialization Finished')

# ========================================
#             Gradio Setting
# ========================================

def gradio_reset(chat_state, img_list):
    if chat_state is not None:
        chat_state.messages = []
    if img_list is not None:
        img_list = []
    return None, gr.update(value=None, interactive=True), gr.update(placeholder='请先上传图片', interactive=False),gr.update(value="上传图片并开始咨询", interactive=True), chat_state, img_list

def upload_img(gr_img, text_input, chat_state):
    if gr_img is None:
        return None, None, gr.update(interactive=True), chat_state, None
    chat_state = CONV_ZH.copy()
    img_list = []
    llm_message = chat.upload_img(gr_img, chat_state, img_list)
    return gr.update(interactive=False), gr.update(interactive=True, placeholder='输入问题'), gr.update(value="开始对话", interactive=False), chat_state, img_list

def gradio_ask(user_message, chatbot, chat_state):
    if len(user_message) == 0:
        return gr.update(interactive=True, placeholder='Input should not be empty!'), chatbot, chat_state
    chat.ask(user_message, chat_state)
    chatbot = chatbot + [[user_message, None]]
    return '', chatbot, chat_state


def gradio_answer(chatbot, chat_state, img_list, num_beams, temperature):
    llm_message = chat.answer(conv=chat_state,
                              img_list=img_list,
                              num_beams=num_beams,
                              temperature=temperature,
                              max_new_tokens=300,
                              max_length=2000)[0]
    chatbot[-1][1] = llm_message
    return chatbot, chat_state, img_list

title = """<h1 align="center"> XrayPULSE </h1>"""
description = """<h3>上传X光影像，开始诊断咨询</h3>"""
disclaimer = """ 
            <h1 >使用说明:</h1>
            <ul> 
                <li>XrayPULSE为PULSE在医疗多模态领域的扩展应用之一，可以用于对X光影像进行医学诊断分析，辅助医生，并为患者提供诊断支持。</li>
                <li>XrayPULSE尝试通过分析X光影像提供准确和有用的结果。然而，我们对所提供结果的有效性、可靠性或完整性不作任何明确的保证或陈述。我们需要不断改善和完善服务，为医疗专业人员提供最好的协助</li>
            </ul>
            <hr> 
            <h3 align="center">OpenMedLab</h3>

            """

def set_example_xray(example: list) -> dict:
    return gr.Image.update(value=example[0])


def set_example_text_input(example_text: str) -> dict:
    return gr.Textbox.update(value=example_text[0])

#TODO show examples below

with gr.Blocks() as demo:
    gr.Markdown(title)
    gr.Markdown(description)

    with gr.Row():
        with gr.Column(scale=0.5):
            image = gr.Image(type="pil")
            upload_button = gr.Button(value="上传影像并开始咨询", interactive=True, variant="primary")
            clear = gr.Button("重制")
            
            num_beams = gr.Slider(
                minimum=1,
                maximum=10,
                value=1,
                step=1,
                interactive=True,
                label="beam search numbers",
            )
            
            temperature = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=1.0,
                step=0.1,
                interactive=True,
                label="Temperature",
            )

        with gr.Column():
            chat_state = gr.State()
            img_list = gr.State()
            chatbot = gr.Chatbot(label='XrayPULSE')
            text_input = gr.Textbox(label='用户', placeholder='请上传X光影像', interactive=False)


    with gr.Row():
        example_xrays = gr.Dataset(components=[image], label="X光影像范例",
                                    samples=[
                                        [os.path.join(os.path.dirname(__file__), "images/example_images/img1.png")],
                                        [os.path.join(os.path.dirname(__file__), "images/example_images/img2.png")],
                                        [os.path.join(os.path.dirname(__file__), "images/example_images/img3.png")],
                                        [os.path.join(os.path.dirname(__file__), "images/example_images/img4.png")],
                                        [os.path.join(os.path.dirname(__file__), "images/example_images/img5.png")],
                                        [os.path.join(os.path.dirname(__file__), "images/example_images/img6.png")],
                                        [os.path.join(os.path.dirname(__file__), "images/example_images/img7.png")],
                                        [os.path.join(os.path.dirname(__file__), "images/example_images/img8.png")],
                                        [os.path.join(os.path.dirname(__file__), "images/example_images/img9.png")],
                                    ])
        

    with gr.Row():
        example_texts = gr.Dataset(components=[gr.Textbox(visible=False)],
                                    label="咨询问题范例",
                                    samples=[
                                        ["详细描述所给的胸部X光影像。"],
                                        ["请观察这张胸部X光影像，并阐述你的发现和总结。"],
                                        ["你能否对所给的胸部X光影像进行详细的描述？"],
                                        ["尽可能详细地描述所给的胸部X光影像。"],
                                        ["这张胸部X光影像中的关键症状是什么？"],
                                        ["你能在这张胸部X光影像中，指出存在的任何异常或需要注意的地方吗"],
                                        ["这张胸部X光影像中，有哪些肺部和心脏的具体特征可见？"],
                                        ["在这张胸部X光影像中，最显著的特征是什么，它是如何反映出病人的健康状况？"],
                                        ["根据从这张胸部X光影像中观察到的发现，给出影像的总体印象是正常还是异常？"],
                                    ],)
    
    example_xrays.click(fn=set_example_xray, inputs=example_xrays, outputs=example_xrays.components)

    upload_button.click(upload_img, [image, text_input, chat_state], [image, text_input, upload_button, chat_state, img_list])
    
    click_response = example_texts.click(set_example_text_input, inputs=example_texts, outputs=text_input).then(
        gradio_ask, [text_input, chatbot, chat_state], [text_input, chatbot, chat_state], queue=False)
    click_response.then(
        gradio_answer, [chatbot, chat_state, img_list, num_beams, temperature], [chatbot, chat_state, img_list], queue=False
    )
    
    submit_response = text_input.submit(gradio_ask, [text_input, chatbot, chat_state], [text_input, chatbot, chat_state], queue=False)
    submit_response.then(
        gradio_answer, [chatbot, chat_state, img_list, num_beams, temperature], [chatbot, chat_state, img_list], queue=False
    )
    clear.click(gradio_reset, [chat_state, img_list], [chatbot, image, text_input, upload_button, chat_state, img_list], queue=False)
    
    gr.Markdown(disclaimer)
demo.launch(share=True, enable_queue=True)
