import glob
import os
from pdf2image import convert_from_path

def convert_pdf_to_images(input_path: str, output_dir: str):
    pdf_paths = glob.glob(os.path.join(input_path, "*.pdf"))
    for pdf_path in pdf_paths:
        images = convert_from_path(pdf_path)
        for i, image in enumerate(images):
            image.save(os.path.join(output_dir, f"{os.path.basename(pdf_path)}_{i}.png"))



