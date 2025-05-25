import os
from src.datasets import M3DocVQA
from src.db import chroma_client
import numpy as np
from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.document import DocTagsDocument
from PIL import Image
from src.db.milvus import MilvusColbertRetriever
from deepeval import evaluate
from deepeval.metrics import MultimodalContextualPrecisionMetric
from deepeval.test_case import MLLMTestCase, MLLMImage

metric = MultimodalContextualPrecisionMetric()
test_case = MLLMTestCase(
    input=["Tell me about some landmarks in France"],
    actual_output=[
        "France is home to iconic landmarks like the Eiffel Tower in Paris.",
    ],
    expected_output=[
        "The Eiffel Tower is located in Paris, France.",
    ],
    retrieval_context=[
        "The Eiffel Tower is a wrought-iron lattice tower built in the late 19th century.",
    ],
)

metric.measure(test_case)
print(metric.score)
print(metric.reason)

image_paths = os.listdir("data/images")

