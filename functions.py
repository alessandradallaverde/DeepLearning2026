import torch
import numpy as np
import torch.nn.functional as F
import json

def encode_queries(queries, processor, model):
    """Extrcat text features from queries texts.

    Arg:
    queries: list of queries in the .js file

    Return:
    text_z: list of (normalized) text features
    """
    # tokenize the input queries
    text_t = processor(text=queries, return_tensors="pt", padding=True)

    # input tensors to cuda
    text_ids = text_t['input_ids'].to("cuda")
    # attention mask to cuda
    attention_mask = text_t['attention_mask'].to("cuda")

    # encode the text with CLIP (latent space)
    with torch.no_grad():
        text_z_output = model.get_text_features(
            input_ids=text_ids,
            attention_mask=attention_mask
        )

    text_z = text_z_output.pooler_output # the tensor
    # normalize the features
    text_z = text_z / text_z.norm(dim=-1, keepdim=True)

    return text_z

def get_annotations(annotations_path):
    """Extract queries and groundtruth annotations from .js file.

    Arg:
    annotations_path: path to the .js file

    Return:
    annotations: list of queries and groundtruth annotations
    """
    with open(annotations_path, "r") as f:
        annotations = json.load(f)

    return annotations

def cosine_similarity(image_query_z, image_z):
    """Computes cosine similarity between two images.

    Arg:
        image_query_z: (normalized) image features modified with the query texts
        image_z: (normalized) image features

    Return:
        cosine similarity between the two images
    """
    return F.cosine_similarity(image_query_z.unsqueeze(1), image_z.unsqueeze(0), dim=-1).cpu()

def get_predictions(modified_target, k, data_path):
    """Find the predictions based on cosine similarity.

    Arg:
    modified_target: tensor of the target image modified with the query
    k: the cutoff for top-K evaluation (e.g., 1, 5, 10)
    data_path: path where images features are contained

    Return:
    predictions: dictionary of the first k predictions
    """
    images_sim_cos = {}
    for pt_file in data_path.glob("*.pt"):
        tensor = torch.load(pt_file).to("cuda")
        idx = int(pt_file.name.replace(".pt", ""))
        images_sim_cos[idx] = cosine_similarity(modified_target, tensor)

    predictions = sorted(
    images_sim_cos.items(),
    key=lambda x: x[1],
    reverse=True
    )
    predictions = dict(predictions[:k])

    return predictions

def evaluate_retrieval(
    retrieved_indices: list[int],
    ground_truth_indices: list[int],
    k: int
):
    """
    Evaluate the retrieval performance for a single source image.

    Args:
    ----
        retrieved_indices: list of image IDs predicted by the model,
            ordered by similarity (descending).
        ground_truth_indices: list of valid target IDs from the benchmark JSON.
        k: the cutoff for top-K evaluation (e.g., 1, 5, 10).

    Return:
    ------
        A dictionary containing Recall@K and Precision@K.

    """
    # Isolate the top K predictions
    top_k_retrieved = retrieved_indices[:k]

    # Calculate the intersection between predictions and ground truth
    hits = set(top_k_retrieved).intersection(set(ground_truth_indices))
    num_hits = len(hits)

    # Metrics calculations
    # Recall@K (Hit Rate): 1 if at least one match is found, 0 otherwise
    recall_at_k = 1 if num_hits > 0 else 0

    # Precision@K: Fraction of top K predictions that are correct
    precision_at_k = num_hits / k

    return {
        f"Recall@{k}": recall_at_k,
        f"Precision@{k}": precision_at_k
    }

def get_unsigned_queries(annotations):
    """Creates a list of unsigned queries (ex: +Smiling becomes Smiling)

    Arg:
    annotations: list of queries and groundtruth annotations

    Return:
    queries_unsigned: list of unsigned queries
    """
    queries = [a["query"] for a in annotations]

    queries_unsigned = []
    for query in queries:
    # if a query contains multiple features, divide them
    # and remove sign
        if ',' in query:
            multiple_q = [q.strip() for q in query.split(",")]
            multiple_q = [q[1:] for q in multiple_q]
            queries_unsigned += multiple_q
        else:
            # remove sign (first character)
            queries_unsigned.append(query[1:])

    return list(dict.fromkeys(queries_unsigned))

def modify_target(visual_path, target_idx, texts, query):
    """Apply query to the target image (latent space arithmetic)

    Arg:
    visual_path: path where images features are contained
    target_idx: index of the target image in celeba
    texts: dictionary of unsigned queries and text features
    query: query to be applied

    Return:
    target_tensor: tensor of the target image modified with the query
    """
    # extract visual features of the specific target
    filename = f"{target_idx}.pt"
    file_path = visual_path / filename
    target_tensor = torch.load(file_path, map_location="cpu").to("cuda")


    # perform naive arithmetic operations in the latent space
    multiple_q = [q.strip() for q in query.split(",")]
    for q in multiple_q:
        if '+' in q:
            target_tensor = target_tensor + texts[q[1:]].unsqueeze(0)
        else:
            target_tensor = target_tensor - texts[q[1:]].unsqueeze(0)

    target_tensor = target_tensor / target_tensor.norm(dim=-1, keepdim=True)

    return target_tensor

def test_query(query_id, multiple_k, directory, annotations, texts, n_images = None):
    """Test the model with a specific query.

    Arg:
    query_id: index of the query in the .js file
    multiple_k: list containing the cutoffs for top-K evaluation (e.g., 1, 5, 10)
    n_images: number of reference images to test the model
    directory: path where images features are contained
    annotations: contains the queries and groundtruth
    texts: dictionary of unsigned queries and text features
    """
    # get all reference images for the query passed in input
    target_images_ids = list(annotations[query_id]["ground_truth"].keys())

    # if the number of reference images is not specified, explore all the images
    if n_images is None:
        n_images = len(target_images_ids)

    # mean of evaluation metrics for different k
    total_recall = np.zeros(len(multiple_k))
    total_precision = np.zeros(len(multiple_k))

    # for every reference image
    for id in target_images_ids[:n_images]:
    # fusion mechanism
        modified_target = modify_target(
            directory,
            id,
            texts,
            annotations[query_id]["query"]
        )
        # predictions
        predictions = get_predictions(modified_target, max(multiple_k), directory)
        # store first prediction (necessary to analyze plots)
        if id == target_images_ids[0]:
            first_pred = predictions
        # compute metrics with different k
        for i, k in enumerate(multiple_k):
            eval = evaluate_retrieval(
                list(predictions.keys()),
                list(annotations[query_id]["ground_truth"][id]),
                k
            )
            total_recall[i] += eval[f"Recall@{k}"]
            total_precision[i] += eval[f"Precision@{k}"]

    # compute and print average metrics
    average_recall = total_recall / n_images
    average_precision = total_precision / n_images
    print(f"Query {annotations[query_id]["query"]}:")
    for i, k in enumerate(multiple_k):
        print(f"  Average Recall@{k}: {average_recall[i]:.4f}")
        print(f"  Average Precision@{k}: {average_precision[i]:.4f}")

    return first_pred

def compose_query(query):
    """Method to modify queries before encoding them with CLIP text encoder.

    Arg:
        query: query in the .js file

    Return:
        final_text_input: modified query
    """
    attributes = [a.strip() for a in query.split(",")]
    signs = [a[0] for a in attributes]
    unsigned_attributes = [a[1:] for a in attributes]
    unsigned_attributes = [a.replace("_", " ").lower() for a in unsigned_attributes]

    idx = 0
    final_text_input = ""
    while idx < len(unsigned_attributes):
        match(unsigned_attributes[idx]):
            case "smiling" | "wearing lipstick" | "wearing hat":
                text_sign =  "is " if signs[idx] == "+" else "is not "
            case "male" | "young":
                text_sign =  "" if signs[idx] == "+" else "not "
            case _:
                text_sign =  "with " if signs[idx] == "+" else "without "

        final_text_input += f"{text_sign}{unsigned_attributes[idx]}"

        idx = idx + 1
        if idx < len(unsigned_attributes):
            final_text_input += " and "

    return final_text_input

def text_arithmetic(texts, query):
    """Apply query to the target image (latent space arithmetic)

    Arg:
    visual_path: path where images features are contained
    target_idx: index of the target image in celeba
    texts: dictionary of unsigned queries and text features
    query: query to be applied

    Return:
    target_tensor: tensor of the target image modified with the query
    """


    # perform naive arithmetic operations in the latent space
    multiple_q = [q.strip() for q in query.split(",")]
    first_q = multiple_q[0]
    target_tensor = torch.zeros_like(texts[first_q[1:]].unsqueeze(0))
    for q in multiple_q:
        if '+' in q:
            target_tensor += texts[q[1:]].unsqueeze(0)
        else:
            target_tensor -= texts[q[1:]].unsqueeze(0)

    target_tensor = target_tensor / target_tensor.norm(dim=-1, keepdim=True)

    return target_tensor

def slerp(visual_path, target_idx, w, alpha):
    """Slerp implementation.

    Arg:
        visual_path: directory path to the frozen visual features
        target_idx: idx of the refrence image
        w: text features of the query
        alpha: hyperparameter of slerp
    """
    # extract visual features of the specific target
    filename = f"{target_idx}.pt"
    file_path = visual_path / filename
    v = torch.load(file_path, map_location="cpu").to("cuda")
    w = w.to("cuda")

    assert v.shape == w.shape, "shapes of v0 and v1 must match"

    v_w_dot = (v * w).sum(-1, keepdim=True)
    v_w_dot = torch.clamp(v_w_dot, -1.0, 1.0)
    theta = torch.acos(v_w_dot)

    # needed to avoid denominator near to 0
    eps = 1e-6

    first_arg = torch.sin((1 - alpha) * theta) / torch.sin(theta + eps)
    second_arg = torch.sin(alpha * theta) / torch.sin(theta + eps)

    v_target = first_arg * v + second_arg * w

    v_target = F.normalize(v_target, dim=-1)

    return v_target

def test_query_slerp(query_id, multiple_k, directory, alpha, annotations, processor, model, n_images = None, texts = None):
    """Test the model with a specific query.

    Arg:
        query_id: index of the query in the .js file
        k: list containing the cutoff for top-K evaluation (e.g., 1, 5, 10)
        n_images: number of reference images to test the model
        directory: path where images features are contained
        annotations: contains the queries and groundtruth
    """
    target_images_ids = list(annotations[query_id]["ground_truth"].keys())

    # if the number of reference images is not specified, explore all the images
    if n_images is None:
        n_images = len(target_images_ids)

    # mean of evaluation metrics for different k
    total_recall = np.zeros(len(multiple_k))
    total_precision = np.zeros(len(multiple_k))

    for id in target_images_ids[:n_images]:
        # modify query before computing the text features with CLIP text encoder
        if texts is None:
            modified_query = compose_query(annotations[query_id]["query"])
            query_t_features = encode_queries(modified_query, processor, model)
        else: # latent space arithmetic
            query_t_features = text_arithmetic(texts, annotations[query_id]["query"])

        v_target = slerp(directory, id, query_t_features, alpha)
        predictions = get_predictions(v_target, max(multiple_k), directory)

        if id == target_images_ids[0]:
            first_pred = predictions

        for i, k in enumerate(multiple_k):
            eval = evaluate_retrieval(
                list(predictions.keys()),
                list(annotations[query_id]["ground_truth"][id]),
                k
            )
            total_recall[i] += eval[f"Recall@{k}"]
            total_precision[i] += eval[f"Precision@{k}"]

    # compute and print average metrics
    average_recall = total_recall / n_images
    average_precision = total_precision / n_images
    print(f"Query {annotations[query_id]["query"]}:")
    for i, k in enumerate(multiple_k):
        print(f"  Average Recall@{k}: {average_recall[i]:.4f}")
        print(f"  Average Precision@{k}: {average_precision[i]:.4f}")

    return first_pred