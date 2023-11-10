from transformers import AutoTokenizer, CLIPTextModel
import torch
from torch.nn import CosineSimilarity
from loguru import logger

prompt_with_type = {
    "describe the pic": "sam",
    "describe the image": "sam",
    "describe the picture": "sam",
    "describe the photo": "sam",
    "segment a": "sam",
    "segment the": "sam",
    "create a picture": "sd",
    "generate a picture": "sd",
    "generate a picture of": "sd",
    "segment all the object": "openseed",
    "segment everything": "openseed",
    "panoptic segment": "openseed",
    "panoptic segment the": "openseed",
}


class DecoderRouter:
    def __init__(self) -> None:
        self.cossim = CosineSimilarity(dim=0, eps=1e-6)

    def calc_similarity(self, v1, v2):
        return self.cossim(v1, v2)

    def init_model(self, model_name_or_path: str = "openai/clip-vit-large-patch14"):
        self.model = CLIPTextModel.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    def need_run(self, prompt, type):
        prompts = [prompt]
        prompts.extend(prompt_with_type.keys())
        inputs = self.tokenizer(
            prompts,
            padding=True,
            return_tensors="pt",
        )

        outputs = self.model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        flattened_hidden_state = torch.flatten(last_hidden_state, 1, -1)
        others = prompts[1:]
        max_idx = 0
        max_score = 0
        for idx, cmp_prompt in enumerate(others):
            sim = self.calc_similarity(
                flattened_hidden_state[0], flattened_hidden_state[idx + 1]
            )
            if sim > max_score:
                max_score = sim
                max_idx = idx + 1
            # logger.info(f"{prompt} <-> {cmp_prompt} = {sim:.4f}")
        best_prompt = prompts[max_idx]
        logger.info("best match prompt {}", best_prompt)
        return prompt_with_type[best_prompt] == type


decoder_router = DecoderRouter()

if __name__ == "__main__":
    # TODO should call from outside
    decoder_router.init_model(
        model_name_or_path="/opt/product/LLaVA/checkpoints/clip-vit-large-patch14"
    )

    prompt = "segment the image"
    type = "sam"
    print(decoder_router.need_run(prompt, type))

    prompt = "panoptic segment"
    print(decoder_router.need_run(prompt, "sd"))
