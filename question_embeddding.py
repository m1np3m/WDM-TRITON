import numpy as np
import torch
from transformers import ColPaliForRetrieval, ColPaliProcessor
from PIL import Image
import jsonlines

QUESTION_PATH = "m3docvqa/multimodalqa/MMQA_dev.jsonl"
QUESTION_EMBEDDING_PATH = ""

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "vidore/colpali-v1.2-hf"

processor = ColPaliProcessor.from_pretrained(model_name)

model = ColPaliForRetrieval.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map=device,
).eval()

with jsonlines.open(QUESTION_PATH) as reader:
    for line in reader:
        question_id = line["qid"]
        question = line["question"]

        inputs = processor(text=question, return_tensors="pt").to(device)
        with torch.no_grad():
            text_embeds = model(**inputs, return_tensors="pt").embeddings
        
        question_embedding = text_embeds.to(torch.float32).detach().cpu().numpy()

        np.save(QUESTION_EMBEDDING_PATH + f"{question_id}.npy", question_embedding)
        

