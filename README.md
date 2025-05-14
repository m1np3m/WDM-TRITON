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

### Create Milvus DB
```bash
python create_milvus_db.py
```

## Evaluation
- Answer Relevancy
- Faithfulness
- Contextual Precision
- Contextual Recall
- Contextual Relevancy

```bash
python eval.py --question_embedding_folder "m3docvqa/question_embeddings" --qa_file "m3docvqa/multimodalqa/MMQA_dev.jsonl" --num_question 5 --image_folder "m3docvqa/images_dev" --db "milvus" --topk 5 --output_file "eval_results.jsonl"
```
## Demo
- Milvus DB
```bash
python demo.py --question_embedding_folder "m3docvqa/question_embeddings" --qa_file "m3docvqa/multimodalqa/MMQA_dev.jsonl" --num_question 5 --image_folder "m3docvqa/images_dev" --db "milvus" --topk 5 --output_file "demo_results.jsonl"
```

- Chroma DB
```bash
python demo.py --question_embedding_folder "m3docvqa/question_embeddings" --qa_file "m3docvqa/multimodalqa/MMQA_dev.jsonl" --num_question 5 --image_folder "m3docvqa/images_dev" --db "chroma" --topk 5 --output_file "demo_results.jsonl"
```




