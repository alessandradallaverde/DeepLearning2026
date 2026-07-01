from pathlib import Path
from torchvision.datasets import CelebA
from transformers import CLIPModel, CLIPProcessor

from functions import *


celeba = CelebA(root=Path("data/celeba"), split="test", download=False)

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = model.cuda().eval()

directory = Path("data/frozen_data")
annotations_path = Path(
    "data/celeba_evaluation.json"
)
annotations = get_annotations(annotations_path)
image_features = get_frozen_image_features(directory)
for k in image_features:
	image_features[k] = image_features[k].to("cuda")

# BASELINE SETUP
# get queries
queries_unsigned = get_unsigned_queries(annotations)
# obtain text features of the queries
text_features = encode_queries(queries_unsigned, processor, model)
# store mapping between text features and unsigned queries
texts = dict(zip(queries_unsigned, text_features))

for query_id in range(len(annotations)):
  print(f"Baseline results ({annotations[query_id]['query']}):")
  _ = test_query(query_id, [1, 5, 10], directory, annotations, texts, n_images=10, image_features=image_features)
  print(f"Slerp results ({annotations[query_id]['query']}):")
  _ = test_query_slerp(query_id, [1, 5, 10], directory, 0.85, annotations, processor, model, n_images=10, texts=None, image_features=image_features)
  print(f"Slerp arithmetic results ({annotations[query_id]['query']}):")
  _ = test_query_slerp(query_id, [1, 5, 10], directory, 0.85, annotations,processor, model, n_images=10, texts=texts, image_features=image_features)
