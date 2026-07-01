
from pathlib import Path
from torchvision.datasets import CelebA
from transformers import CLIPModel, CLIPProcessor

from DeepLearning2026.functions import *


celeba = CelebA(root=Path("data/celeba"), split="test", download=False)

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = model.cuda().eval()

directory = Path("data/frozen_data")
annotations_path = Path(
    "data/celeba_evaluation.json"
)
annotations = get_annotations(annotations_path)

# BASELINE SETUP
# get queries
queries_unsigned = get_unsigned_queries(annotations)
# obtain text features of the queries
text_features = encode_queries(queries_unsigned, processor, model)
# store mapping between text features and unsigned queries
texts = dict(zip(queries_unsigned, text_features))

for query_id in range(1, len(annotations)):
  print(f"Baseline results ({annotations[query_id]['query']}):")
  _ = test_query(query_id, [1, 5, 10], directory, annotations, texts, None)
  print(f"Slerp results ({annotations[query_id]['query']}):")
  _ = test_query_slerp(query_id, [1, 5, 10], directory, 0.85, annotations, None)
  print(f"Slerp arithmetic results ({annotations[query_id]['query']}):")
  _ = test_query_slerp(query_id, [1, 5, 10], directory, 0.85, annotations, None, texts)