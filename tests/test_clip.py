from transformers import AutoTokenizer, CLIPTextModel
import torch
from torch.nn import CosineSimilarity


def test_clip_text_embedding_similarity():
    model_name_or_path = "/opt/product/LLaVA/checkpoints/clip-vit-large-patch14"
    model = CLIPTextModel.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    prompts = [
        "panoptic segment the women",
        "segment a",
        "segment the",
        "segment all the object",
        "segment everything",
        "panoptic segment",
    ]
    inputs = tokenizer(
        prompts,
        padding=True,
        return_tensors="pt",
    )

    cossim = CosineSimilarity(dim=0, eps=1e-6)

    def dist(v1, v2):
        return cossim(v1, v2)

    outputs = model(**inputs)
    last_hidden_state = outputs.last_hidden_state
    flattened_hidden_state = torch.flatten(last_hidden_state, 1, -1)
    pooled_output = outputs.pooler_output  # pooled (EOS token) states
    print("pooled_output.shape", pooled_output.shape)
    for i1, label1 in enumerate(prompts):
        for i2, label2 in enumerate(prompts):
            if i2 >= i1:
                print(
                    f"{label1} <-> {label2} = {dist(flattened_hidden_state[i1], flattened_hidden_state[i2]):.4f}"
                )

    # print("=================================================")
    # for i1, label1 in enumerate(prompts):
    #     for i2, label2 in enumerate(prompts):
    #         if i2 >= i1:
    #             print(
    #                 f"{label1} <-> {label2} = {dist(pooled_output[i1], pooled_output[i2]):.4f}"
    #             )
