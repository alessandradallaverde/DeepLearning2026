import torch
from pathlib import Path
from torchvision.datasets import CelebA
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torchvision.transforms as transforms

def get_celeba(root, batch_size = 1024, download = False):
    transform = transforms.ToTensor()

    training_data = CelebA(root=root, split="train", download=False, transform=transform)
    validation_data = CelebA(root=root, split="valid", download=False, transform=transform)
    test_data = CelebA(root=root, split="test", download=False, transform=transform)

    print(f"# of training samples: {len(training_data)}")
    print(f"# of validation samples: {len(validation_data)}")
    print(f"# of test samples: {len(test_data)}")

    train_loader = torch.utils.data.DataLoader(training_data, batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(validation_data, batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def get_image_features(clip_model, clip_processor, images, device = "cuda"):
    batch_features = clip_processor(
        images=images,
        return_tensors="pt",
        padding = True
    )
    # pixel values of the images
    pixel_values = batch_features.pixel_values.to(device)

    # encode the images with CLIP
    with torch.no_grad():
        images_z_output = clip_model.get_image_features(pixel_values=pixel_values)

    images_z = images_z_output.pooler_output # the tensors

    return images_z

def get_normalized_image_features(images_features):
    images_z = images_features / images_features.norm(dim=-1, keepdim=True)

    return images_z

def encode_text_with_image(model, input_ids, img_tokens):
    """
    Injects an image token into CLIP text embeddings just before the End-of-Text (EOT) token.

    Args:
        model: a frozen Hugging Face CLIPModel (e.g., 'openai/clip-vit-base-patch32').
        input_ids: text token IDs, shape [batch_size, n_ctx] (typically n_ctx=77).
        img_tokens: image features to inject, shape [batch_size, hidden_size].

    Returns:
        Projected multimodal features, shape [batch_size, projection_dim].
    """
    # 1. Ensure the model is in evaluation mode (frozen)
    # TODO: change this because it in ensured in the training loop
    model.eval()
    batch_size = input_ids.size(0)

    with torch.no_grad():
        # get the base token embeddings from CLIP
        token_embeds = model.text_model.embeddings.token_embedding(input_ids)

    # find the EOT token index
    eot_indices = input_ids.argmax(dim=-1)

    # take the join index from the first item in the batch
    splice_idx = eot_indices[0].item()

    # insert the image tokens into the embeddings
    img_tokens = img_tokens.view(batch_size, 1, -1)
    spliced_embeds = torch.cat([
        token_embeds[:, :splice_idx],
        img_tokens,
        token_embeds[:, splice_idx:-1]
    ], dim=1)

    # create a runtime patch for the embedding layer's forward method
    # while CLIPTextModel ignores inputs_embeds, its underlying
    # CLIPTextEmbeddings layer accepts it
    original_embeddings_forward = model.text_model.embeddings.forward

    def patched_embeddings_forward(
        input_ids=None,
        position_ids=None,
        inputs_embeds=None
    ):
        # we intercept the call, drop input_ids, and force feed our custom
        # token embeddings
        return original_embeddings_forward(
            input_ids=None,
            position_ids=position_ids,
            inputs_embeds=spliced_embeds
        )

    try:
        # apply the runtime patch
        model.text_model.embeddings.forward = patched_embeddings_forward

        # run high-level text model safely
        outputs = model.text_model(input_ids=input_ids)
        last_hidden_state = outputs.last_hidden_state

    finally:
        # restore original method to avoid breaking downstream text tracking
        model.text_model.embeddings.forward = original_embeddings_forward

    # extract features at the shifted EOS token position
    # (+1 due to the injection)
    new_eot_indices = eot_indices + 1
    batch_indices = torch.arange(batch_size, device=input_ids.device)
    eot_features = last_hidden_state[batch_indices, new_eot_indices]

    projected_features = model.text_projection(eot_features)

    return projected_features

def get_normalized_text_features(clip_model, clip_processor, token_features, device = "cuda"):
    # tokenize text
    text_inputs = clip_processor(
        text="a photo of",
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt"
    )

    # extract input_ids
    input_ids = text_inputs["input_ids"].to(device)

    # repeat the tokenized row to match the batch size of token_features
    input_ids = input_ids.repeat(token_features.size(0), 1)  # Shape: [batch_size, 77]

    # features + pseudo language token
    text_features = encode_text_with_image(clip_model, input_ids, token_features)

    # normalize the features
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return text_features

class IM2TEXT(nn.Module):
    def __init__(self, embed_dim=512, middle_dim=512, output_dim=512, n_layer=2, dropout=0.1):
        super().__init__()
        self.fc_out = nn.Linear(middle_dim, output_dim)
        layers = []
        dim = embed_dim
        for _ in range(n_layer):
            block = []
            block.append(nn.Linear(dim, middle_dim))
            block.append(nn.Dropout(dropout))
            block.append(nn.ReLU())
            dim = middle_dim
            layers.append(nn.Sequential(*block))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return self.fc_out(x)
    
def get_optimizer(img2text, wd = 0.02, lr = 1e-4):
    named_parameters = list(img2text.named_parameters())
    exclude = lambda n : "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
    include = lambda n : not exclude(n)
    gain_or_bias_params = [p for n, p in named_parameters if exclude(n) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(n) and p.requires_grad]

    optimizer = torch.optim.AdamW(
        [
            {"params": gain_or_bias_params, "weight_decay": 0.0},
            {"params": rest_params, "weight_decay": wd},
        ],
        lr=lr,
    )

    return optimizer

def get_loss_img2text(clip_model, clip_processor, loss_img, loss_txt, images_features, token_features, device = "cuda"):

    image_features = get_normalized_image_features(images_features)
    text_features = get_normalized_text_features(clip_model, clip_processor, token_features)

    logit_scale = clip_model.logit_scale.exp()
    logit_scale = logit_scale.mean()

    ground_truth = torch.arange(len(image_features)).long()
    ground_truth = ground_truth.to(device)

    logits_per_image = logit_scale * image_features @ text_features.t()
    loss_img_val = loss_img(logits_per_image, ground_truth)
    logits_per_text = logit_scale * text_features @ image_features.t()
    loss_txt_val = loss_txt(logits_per_text, ground_truth)
    total_loss = (loss_img_val + loss_txt_val) / 2

    return image_features, text_features, total_loss

def cosine_similarity_mean(text_pseudo_z, image_z):
    return F.cosine_similarity(text_pseudo_z.unsqueeze(1), image_z.unsqueeze(0), dim=-1).mean().cpu()

def cosine_similarity(text_pseudo_z, image_z):
    return F.cosine_similarity(text_pseudo_z.unsqueeze(1), image_z.unsqueeze(0), dim=-1)

def training_step(
    clip_model,
    clip_processor,
    img2text,
    training_data_loader,
    optimizer,
    loss_img,
    loss_txt,
    epoch,
    device="cuda",
    tb_writer=None
):
    num_batches_per_epoch = len(training_data_loader)
    total_cossim = 0
    # clip model not in train mode (pretrained and frozen)
    clip_model.eval()
    # set the network to training mode
    img2text.train()

    # iterate over the training set
    for i, (images, texts) in enumerate(training_data_loader):
        # store the actual step of the training
        step = num_batches_per_epoch * epoch + i

        # reset gradients
        optimizer.zero_grad()

        # move input images to cuda
        images = images.to(device)
        # extract CLIP features from the raw images
        clip_image_features = get_image_features(
            clip_model,
            clip_processor,
            images, device
        )
        # forward pass: img2text processes the CLIP image features
        img_tokens = img2text(clip_image_features)
        # loss computation: pass raw images and img_tokens to the loss function
        image_features, text_features, loss = get_loss_img2text(
            clip_model,
            clip_processor,
            loss_img,
            loss_txt,
            clip_image_features,
            img_tokens,
            device
        )

        # backward pass (compute the gradients)
        loss.backward()

        # parameters update
        optimizer.step()

        total_cossim += cosine_similarity_mean(text_features, image_features)

        # every 64 samples print training evolution
        if (i%64) == 0:
            num_samples = i * len(images)
            samples_per_epoch = len(training_data_loader.dataset)
            percent_complete = 100.0 * i / num_batches_per_epoch
            # print status of training
            print(
                f"Train Epoch: {epoch} [{num_samples}/{samples_per_epoch} ({percent_complete:.0f}%)]\t"
                f"Loss: {loss.item():.6f}"
                f"\tLR: {optimizer.param_groups[0]['lr']:5f}\tlogit_scale {clip_model.logit_scale.data:.3f}"
            )

        # every 64 samples store loss updates (also scale and lr, that for now
        # are static) on the tensor board
        if (i%32) == 0:
            timestep = epoch * num_batches_per_epoch + i
            log_data = {
                "loss": loss.item(),
                "scale":  clip_model.logit_scale.data.item(),
                "lr": optimizer.param_groups[0]["lr"],
                "cossim": cosine_similarity_mean(text_features, image_features)
            }

            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, timestep)

    total_cossim = total_cossim/len(training_data_loader.dataset)

def main(
    clip_model,
    clip_processor,
    exp_name="exp1",
    root=Path("/content/datasets"),
    batch_size=128,     # how many samples to process in parallel. GPU-dependent
    device="cuda",    # Where to perform calculations
    wd = 0.02,
    lr = 1e-4,
    epochs=2,          # How many times to iterate over the entire train set
):

    # create a logger for the experiment
    writer = SummaryWriter(log_dir=f"runs/{exp_name}")

    # get dataloaders
    train_loader, val_loader, test_loader = get_celeba(batch_size, root)

    # frozen CLIP model
    clip_model = clip_model.cuda().eval()

    # instantiate the network and move it to the chosen device (GPU)
    img2text = IM2TEXT().to(device)

    # "print" the network to view all the modules, so its architecture
    print(img2text)

    # instantiate the optimizer
    optimizer = get_optimizer(img2text, wd, lr)

    # create loss functions and move them to the GPU
    loss_img = nn.CrossEntropyLoss().to(device)
    loss_txt = nn.CrossEntropyLoss().to(device)

    # for each epoch, train the network and then compute evaluation results
    for e in range(epochs):
        total_cosim = training_step(
            clip_model,
            clip_processor,
            img2text,
            train_loader,
            optimizer,
            loss_img,
            loss_txt,
            e,
            device="cuda",
            tb_writer=writer
        )
        print(f"Total cossim (epoch {e}): {total_cosisim}")

    # closes the logger
    writer.close()
    # store the trained model
    torch.save(img2text.state_dict(), f"model_{exp_name}.pth")

    return img2text