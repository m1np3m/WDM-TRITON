# TRITON-TEAM
Repo for Triton team.

## Install dependencies
```bash
pip install -r requirements.txt
```

## Prepare resources

1. Download the resources from https://drive.google.com/drive/folders/1-GjtJ3N-nAXXZci3ZtbiNGwV_i1KLQ6-
2. Unzip `m3docvqa.zip` and put the `m3docvqa` folder in the root directory.
3. Unzip `embeddings_dev.zip` and `question_embeddings.zip` and put them in the `m3docvqa` folder.

## Database
### Create Chroma DB
```bash
python create_chroma_db.py
```

## Run Demo
```bash
python demo.py --question_embedding_folder m3docvqa/question_embeddings --qa_file m3docvqa/m3docvqa_dev.jsonl --num_question 10 --image_folder m3docvqa/images_dev --output_folder m3docvqa/output
```


