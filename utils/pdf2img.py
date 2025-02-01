import fitz
from PIL import Image
import os


# pdf转image
def pdf_to_image(pdf_path):
    images = []
    resolution = 2048
    with fitz.open(pdf_path) as pdf:
        for page in pdf:
            if page.rect.width < page.rect.height:
                page.set_rotation(270)
            zoom_x = resolution / page.rect.width
            zoom_y = resolution / page.rect.height
            zoom = min(zoom_x, zoom_y)
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
            # if pix.width > 2000 or pix.height > 2000:
            #     pix = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
            img = Image.frombytes('RGB', (pix.width, pix.height), pix.samples)
            images.append(img)
    return images


# pdf转image并保存
def save_pdf_to_image(pdf_path, output_dir, img_format='png'):
    images = pdf_to_image(pdf_path)
    for i, img in enumerate(images):
        img.save(f'{output_dir}/{i + 1}.{img_format}')


if __name__ == '__main__':
    pdf_path = "./电气主接线图.pdf"
    output_dir = "./output"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    save_pdf_to_image(pdf_path, output_dir)
