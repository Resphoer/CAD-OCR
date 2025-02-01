import base64
from http import HTTPStatus

import gradio as gr
import numpy as np
from PIL import Image
from dashscope import Application
from gradio_client import Client, handle_file
from openai import OpenAI
from paddleocr import PaddleOCR, draw_ocr

from api.file_api.upload_file import upload_file
from utils.extract_utils import *
from utils.pdf2img import pdf_to_image

# 全局变量
# PaddleOCR模块
ocr = PaddleOCR(use_angle_cls=True, det_limit_side_len=2200,
                det_model_dir='./weights/ch_PP-OCRv4_det_server_infer',
                rec_model_dir='./weights/ch_PP-OCRv4_rec_server_infer',
                cls_model_dir='./weights/cls_infer/ch_ppocr_mobile_v2.0_cls_slim_infer')
# 前端电缆规格载流量映射
frontend_voltage_mapping = {25: 90, 35: 110, 50: 125, 70: 152, 95: 182, 120: 205, 150: 223, 185: 252, 240: 292,
                            300: 332,
                            400: 378, 500: 428}
# 后端电缆规格载流量映射
backend_voltage_mapping = {25: 87, 35: 105, 50: 123, 70: 148, 95: 178, 120: 200, 150: 232, 185: 262, 240: 300, 300: 343,
                           400: 386, 500: 432}
# todo 通义千问大模型API_KEY
api_key = "YOUR_API_KEY"
app_id = "YOUR_APP_ID"
# 框图名称
frame_label = '电气主接线图'
# 结果
file_id = None
session_id = None
chat_response = None
gallery_images = []
button_choice = None
page_num = None
all_result = []
result_lines = []
result_img = None


# ------------供电图纸OCE识别--------------
# 获取OCR结果
def get_ocr_result(img):
    img_np = np.array(img)
    result = ocr.ocr(img_np, cls=True)
    return result[0] if result else []


# 检查关键字是否在图像的右下角
def is_keyword_in_bottom(image, keywords):
    global frame_label
    img_np = np.array(image)
    h, w, _ = img_np.shape
    # 右下角
    bottom_right_region = img_np[int(h * 0.75):h, int(w * 0.75):w]
    result = ocr.ocr(bottom_right_region, cls=True)
    if not result or not result[0]:
        return False
    for line in result[0]:
        for keyword in keywords:
            if keyword in line[1][0]:
                frame_label = keyword
                return True
    # 左下角
    bottom_left_region = img_np[int(h * 0.75):h, 0:int(w * 0.25)]
    result = ocr.ocr(bottom_left_region, cls=True)
    if not result or not result[0]:
        return False
    for line in result[0]:
        for keyword in keywords:
            if keyword in line[1][0]:
                frame_label = keyword
                return True
    return False


# 查找目标关键字页面
def find_target_page(images, keywords, progress=gr.Progress()):
    global all_result
    all_result = []
    progress(0.5, "提取相关设计图信息中")
    for i, img in enumerate(images):
        if is_keyword_in_bottom(img, keywords):
            ocr_result = get_ocr_result(img)
            all_result.append([i + 1, img, ocr_result])


# 正则匹配
def regex_match(pattern, result_lines):
    result = []
    for line in result_lines:
        match = re.search(pattern, line[1][0])
        if match:
            result.append(line)
    device_positions = [(device[0][0][0], device[1][0]) for device in result]
    device_positions.sort()
    return device_positions[0][1] if device_positions else '', result


# 获取电容器信息
def extract_capacitor_info(progress=gr.Progress()):
    global button_choice
    button_choice = 2
    filter_result = []
    progress(0, '获取电容器柜容量及分补、共补信息中')
    # 电容器
    capacitor_pattern = r'^(\d+).*(kvar|kVar)$'
    capacitor, capacitor_all = regex_match(capacitor_pattern, result_lines)

    if len(capacitor_all) == 0:
        return '未获取到电容器柜容量及分补、共补信息', None, gr.Radio(list(range(1, len(all_result) + 1)), value=1,
                                                    label='施工图页面选择'), gr.update(
            visible=True)

    filter_result.extend(capacitor_all)
    capacitor_capacity = extract_capacitor_capacity(capacitor)

    print(f'电容器柜容量：{capacitor_capacity}')

    # 变压器规格
    transformer_pattern = r'^[^-(]+-[^-]+-[^-]+[/kVA][^-]+$'
    # 正则匹配变压器规格
    transformer, transformer_all = regex_match(transformer_pattern, result_lines)
    filter_result.extend(transformer_all)
    transformer_capacity = extract_transformer_capacity(transformer)

    print(f'变压器容量：{transformer_capacity}')

    progress(0.25, '正在计算中')

    # 规则校验
    max_capacitor_value = transformer_capacity * 0.2
    capacitor_qualified = '电容器柜总容量合格' if capacitor_capacity >= max_capacitor_value else '电容器柜总容量不合格'

    # 分补、共补
    separated_pattern = r'(分补|(BSMJ|BCMJ)-[^-]+-[^-]+-([^-3]+|3\w+))'
    total_pattern = r'(共补|(BSMJ|BCMJ)-[^-]+-[^-]+-3.$)'

    # 正则匹配分补、共补规格
    separated, separated_all = regex_match(separated_pattern, result_lines)
    print(f'分补规格：{separated}')
    filter_result.extend(separated_all)
    separated_capacity = extract_separated_total_capacity(separated)
    print(f'分补容量：{separated_capacity}')
    max_separated_value = capacitor_capacity * 0.4
    separated_qualified = '分补容量合格' if separated_capacity >= max_separated_value else '分补容量不合格'

    total, total_all = regex_match(total_pattern, result_lines)
    print(f'共补规格：{total}')
    filter_result.extend(total_all)
    total_capacity = extract_separated_total_capacity(total)
    print(f'共补容量：{total_capacity}')

    # 设备信息
    info = (
        f"变压器型号: {transformer}\n"
        f"---------------------------\n"
        f"电容器柜总容量: {capacitor_capacity}\n"
        f"允许电容量: {capacitor_capacity}\n"
        f"最大电容量: {max_capacitor_value:.2f}\n"
        f"{capacitor_qualified}\n"
        f"---------------------------\n"
        f"分补规格: {separated}\n"
        f"允许分补容量: {separated_capacity}\n"
        f"最大分补容量: {max_separated_value:.2f}\n"
        f"{separated_qualified}\n"
        f"---------------------------\n"
        f"共补规格: {total}\n"
        f"共补容量: {total_capacity}"
    )

    progress(0.66, '计算完毕，绘制图像中')

    # 在图片上绘制筛选后的结果
    image = result_img
    boxes = [line[0] for line in filter_result]
    im_show = draw_ocr(np.array(image), boxes, font_path='./fonts/simfang.ttf')
    im_show = Image.fromarray(im_show)

    progress(1, '抽取信息完毕')

    return info, im_show, gr.Radio(list(range(1, len(all_result) + 1)), value=1,
                                   label='获取电容器柜总容量及分补、共补信息：施工图页面选择'), gr.update(
        visible=True)


# 获取变压器信息及前后端电缆规格
def extract_voltage_info(progress=gr.Progress()):
    global button_choice
    button_choice = 1

    progress(0, '抽取变压器及前后端电缆信息中')
    # 判断是否有高压电缆
    backend = frame_label == '电气主接线图'

    # 低压电缆规格
    low_voltage_pattern = r'^[^-]+-[^/]+/[^/]+-[^-]+$'
    # 高压电缆规格
    high_voltage_pattern = r'^[A-Za-z]+-[^-]+/[^-]+$'
    # 变压器规格
    transformer_pattern = r'^[^-(]+-[^-]+-[^-]+[/kVA][^-]+$'

    filter_result = []

    # 正则匹配变压器规格
    transformer, transformer_all = regex_match(transformer_pattern, result_lines)
    filter_result.extend(transformer_all)
    transformer_capacity = extract_transformer_capacity(transformer)

    print(f'变压器规格：{transformer}')
    print(f'变压器容量：{transformer_capacity}')

    progress(0.25, '正在计算中')

    # 处理低压电缆
    # 正则匹配电缆型号
    low_voltage, low_voltage_all = regex_match(low_voltage_pattern, result_lines)
    filter_result.extend(low_voltage_all)
    print(f'低压电缆型号：{low_voltage}')
    # 获取电缆规格
    low_voltage_value = extract_low_voltage_number(low_voltage)
    print(f'低压电缆规格: {low_voltage_value}')
    # 额定载流量
    mapped_low_voltage_value = frontend_voltage_mapping.get(low_voltage_value)
    # 计算最大载流量
    actual_current_low = mapped_low_voltage_value
    max_current_low = transformer_capacity / np.sqrt(3) / 10
    low_voltage_qualified = "电缆线径合格" if max_current_low <= actual_current_low else "电缆线径不合格"

    # 若有高压电缆，则处理高压电缆
    if backend:
        # 正则匹配电缆型号
        high_voltage, high_voltage_all = regex_match(high_voltage_pattern, result_lines)
        filter_result.extend(high_voltage_all)
        print(f'高压电缆型号：{high_voltage}')
        # 获取电缆规格
        high_voltage_value = extract_high_voltage_number(high_voltage)
        print(f'高压电缆规格: {high_voltage_value}')
        # 额定载流量
        mapped_high_voltage_value = frontend_voltage_mapping.get(high_voltage_value)
        # 计算最大载流量
        actual_current_high = mapped_high_voltage_value
        max_current_high = transformer_capacity / np.sqrt(3) / 10
        high_voltage_qualified = "电缆线径合格" if max_current_high <= actual_current_high else "电缆线径不合格"

        # 额定载流量
        mapped_low_voltage_value = backend_voltage_mapping.get(low_voltage_value)
        # 计算最大载流量
        actual_current_low = mapped_low_voltage_value
        max_current_low = transformer_capacity / np.sqrt(3) / 0.38
        low_voltage_qualified = "电缆线径合格" if max_current_low <= actual_current_low else "电缆线径不合格"

        # 设备信息
        info = (
            f"变压器型号: {transformer}\n"
            f"---------------------------\n"
            f"变压器前端电缆规格为: {high_voltage}\n"
            f"允许载流量为: {actual_current_high}\n"
            f"最大载流量为: {max_current_high:.2f}\n"
            f"{high_voltage_qualified}\n"
            f"---------------------------\n"
            f"变压器后端电缆规格为: {low_voltage}\n"
            f"允许载流量为: {actual_current_low}\n"
            f"最大载流量为: {max_current_low:.2f}\n"
            f"{low_voltage_qualified}"
        )
    else:
        info = (
            f"变压器型号: {transformer}\n"
            f"---------------------------\n"
            f"变压器前端电缆规格为: {low_voltage}\n"
            f"允许载流量为: {actual_current_low}\n"
            f"最大载流量为: {max_current_low:.2f}\n"
            f"{low_voltage_qualified}"
        )

    progress(0.66, '计算完毕，绘制图像中')

    # 在图片上绘制筛选后的结果
    image = result_img
    boxes = [line[0] for line in filter_result]
    im_show = draw_ocr(np.array(image), boxes, font_path='./fonts/simfang.ttf')
    im_show = Image.fromarray(im_show)

    progress(1, '抽取信息完毕')

    return info, im_show, gr.Radio(list(range(1, len(all_result) + 1)), value=1,
                                   label='获取变压器及其前端、后端电缆规格：施工图页面选择'), gr.update(
        visible=True)


# ocr识别pdf
def ocr_pdf(pdf_path, progress=gr.Progress()):
    global page_num, result_img, result_lines, page_button
    # 目标关键字
    keywords = ['电气主接线图', '箱变主接线图', '箱变系统图', '箱变一次系统图']

    # pdf转image
    images = pdf_to_image(pdf_path)

    progress(0.20, "正在处理PDF页面")

    # 接线图/相变系统图
    find_target_page(images, keywords)

    if len(all_result) == 0:
        return '未找到施工图页面'

    page_num, result_img, result_lines = all_result[0]

    return 'PDF处理完成'


# 基础分页
def change_page(page):
    global page_num, result_img, result_lines
    page_num, result_img, result_lines = all_result[page - 1]
    return f'PDF第{page_num}页'


# 重载结果
def reload_result():
    txt, img, button, column = None, None, None, None

    if button_choice == 1:
        txt, img, button, column = extract_voltage_info()
    elif button_choice == 2:
        txt, img, button, column = extract_capacitor_info()

    return txt, img


# ------------供电图纸OCE识别--------------


# ------------对比实验---------------------
# pdf图片画廊
def transfer_gallery(pdf_input):
    global gallery_images
    gallery_images = pdf_to_image(pdf_input)
    return gallery_images


# 选中索引
def get_select_index(evt: gr.SelectData):
    return evt.index


# 选中图片路径
def get_select_img(evt: gr.SelectData):
    print(evt.value)
    return evt.value['image']['path'], evt.index


# GOT-OCR引擎
def got_ocr(img_pth):
    client = Client("https://stepfun-ai-got-official-online-demo.ms.show/")
    result = client.predict(
        image=handle_file(img_pth),
        got_mode="plain texts OCR",
        fine_grained_mode="box",
        ocr_color="red",
        ocr_box="Hello!!",
        api_name="/run_GOT"
    )

    return result


# qwen-vl-ocr引擎
def qwen_ocr(img_pth):
    #  读取本地文件，并编码为 BASE64 格式
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    base64_image = encode_image(img_pth)
    client = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(
        model="qwen-vl-ocr",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        # 需要注意，传入BASE64，图像格式（即image/{format}）需要与支持的图片列表中的Content Type保持一致。"f"是字符串格式化的方法。
                        # PNG图像：  f"data:image/png;base64,{base64_image}"
                        # JPEG图像： f"data:image/jpeg;base64,{base64_image}"
                        # WEBP图像： f"data:image/webp;base64,{base64_image}"
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                        "min_pixels": 28 * 28 * 4,
                        "max_pixels": 28 * 28 * 1280
                    },
                    # 为保证识别效果，目前模型内部会统一使用"Read all the text in the image."进行识别，用户输入的文本不会生效。
                    {"type": "text", "text": "Read all the text in the image."},
                ],
            }
        ],
    )
    return completion.choices[0].message.content


# OCR识别
def diff_ocr(img_pth, img_idx, model_idx):
    img_idx = int(img_idx)
    model_idx = int(model_idx)
    img = gallery_images[img_idx]
    img_pth = str(img_pth)
    if model_idx == 1:
        return got_ocr(img_pth)[0]
    elif model_idx == 2:
        return qwen_ocr(img_pth)
    else:
        result = get_ocr_result(img)
        txts = [line[1][0] for line in result]
        return " ".join(txts)


# ------------对比实验---------------------


# ------------大模型对话---------------------

def init_session():
    global chat_response, session_id
    chat_response = Application.call(
        # 若没有配置环境变量，可用百炼API Key将下行替换为：api_key="sk-xxx"。但不建议在生产环境中直接将API Key硬编码到代码中，以减少API Key泄露风险。
        api_key=api_key,
        app_id=app_id,  # 应用ID替换YOUR_APP_ID
        prompt='输出LaTeX公式时，请用$$包括'
    )
    session_id = chat_response.output.session_id


def chat(message, history):
    global chat_response, file_id, session_id
    text, files = message['text'], message['files']
    if len(files) != 0:
        file_id = upload_file(files[0])
    if file_id is not None:
        chat_response = Application.call(
            # 若没有配置环境变量，可用百炼API Key将下行替换为：api_key="sk-xxx"。但不建议在生产环境中直接将API Key硬编码到代码中，以减少API Key泄露风险。
            api_key=api_key,
            app_id=app_id,  # 应用ID替换YOUR_APP_ID
            stream=True,
            prompt=text,
            rag_options={
                "session_file_ids": [file_id],
            },
            session_id=session_id
        )
    else:
        chat_response = Application.call(
            # 若没有配置环境变量，可用百炼API Key将下行替换为：api_key="sk-xxx"。但不建议在生产环境中直接将API Key硬编码到代码中，以减少API Key泄露风险。
            api_key=api_key,
            app_id=app_id,  # 应用ID替换YOUR_APP_ID
            stream=True,
            prompt=text,
            session_id=session_id
        )
    for response in chat_response:
        if response.status_code != HTTPStatus.OK:
            return "很抱歉，我无法解答您的问题"
        session_id = response.output.session_id
        yield response.output.text


if __name__ == '__main__':
    # 创建Gradio界面
    with gr.Blocks(theme=gr.themes.Soft(font_mono=[gr.themes.GoogleFont("Montserrat")])) as demo:
        # OCR识别分页
        with gr.Tab(label='供电图纸OCR识别'):
            gr.Markdown(value="<h1 align='center'>供电图纸OCR识别</h1>", show_label=False)
            gr.Markdown(value="<hr>")
            with gr.Row():
                gr.Interface(ocr_pdf,
                             inputs=gr.File(type="filepath", label="上传PDF文件"),
                             outputs=gr.Textbox(placeholder='此处展示PDF处理进度...'),
                             clear_btn=None,
                             flagging_mode='never',
                             live=True)
            with gr.Row():
                button1 = gr.Button("获取变压器及其前端、后端电缆规格")
                button2 = gr.Button("获取电容器柜总容量及分补、共补信息")

            with gr.Column(visible=False) as button_output:
                page_button = gr.Radio()
                pdf_page = gr.Textbox(label='PDF页码')
                gr.Interface(change_page,
                             inputs=page_button,
                             outputs=pdf_page,
                             clear_btn=None,
                             flagging_mode='never',
                             live=True)
                with gr.Row():
                    info_output = gr.Textbox(label="设备信息")
                    page_output = gr.Image(type="pil", label='施工图')
                    gr.on(
                        triggers=[pdf_page.change],
                        fn=reload_result,
                        inputs=None,
                        outputs=[info_output, page_output]
                    )

            button1.click(extract_voltage_info, inputs=None,
                          outputs=[info_output, page_output, page_button, button_output])
            button2.click(extract_capacitor_info, inputs=None,
                          outputs=[info_output, page_output, page_button, button_output])
        with gr.Tab(label='对比实验'):
            with gr.Row():
                page_choice = gr.Number(visible=False)
                img_path = gr.Textbox(visible=False)
                model1 = gr.Number(visible=False)
                model2 = gr.Number(visible=False)
                with gr.Column(scale=2):
                    file_input = gr.File(type='filepath', label='上传PDF文件')
                    gallery = gr.Gallery(label='页面选择', format='png', columns=4, show_label=True, interactive=False)

                    file_input.upload(transfer_gallery, inputs=file_input, outputs=gallery)
                with gr.Column(scale=1):
                    choice1 = gr.Dropdown(label='模型选择', interactive=True,
                                          choices=['PaddleOCR', 'GOT-OCR', 'QWen-VL-OCR'])
                    button1 = gr.Button('开始OCR识别')
                    output1 = gr.Textbox(label='识别结果')
                with gr.Column(scale=1):
                    choice2 = gr.Dropdown(label='模型选择', interactive=True,
                                          choices=['PaddleOCR', 'GOT-OCR', 'QWen-VL-OCR'])
                    button2 = gr.Button('开始OCR识别')
                    output2 = gr.Textbox(label='识别结果')

                gallery.select(get_select_img, inputs=None, outputs=[img_path, page_choice])
                choice1.select(get_select_index, inputs=None, outputs=model1)
                choice2.select(get_select_index, inputs=None, outputs=model2)
                button1.click(diff_ocr, inputs=[img_path, page_choice, model1], outputs=[output1])
                button2.click(diff_ocr, inputs=[img_path, page_choice, model2], outputs=[output2])
        with gr.Tab(label='大模型对话'):
            init_session()
            gr.Markdown(value="<h1 align='center'>供电图纸智能体</h1>", show_label=False)
            gr.ChatInterface(
                fn=chat,
                multimodal=True,
                chatbot=gr.Chatbot(
                    show_copy_button=True,
                    placeholder="<h1>I'm a helpful AI robot</h1><center>You can ask me any questions.</center>",
                    avatar_images=('./assets/user.png', './assets/robot.png')
                ),
                textbox=gr.MultimodalTextbox(
                    file_types=['image'],
                    placeholder='Type a message...',
                ),
                examples=['现在时间是？', '天气怎么样？', '你好！'],
            )
    # 运行程序
    demo.title = '供电图纸识别'
    demo.queue()
    demo.launch()
