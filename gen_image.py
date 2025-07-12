from diffusers import StableDiffusionPipeline
import torch

# 파이프라인 로드
pipe = StableDiffusionPipeline.from_pretrained(
    "./dreambooth_output",
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

# 이미지 생성
image = pipe(
    "a photo of sks cat wearing a red shirt",
    num_inference_steps=30,
    guidance_scale=7.5
).images[0]

image.save("generated_cat.png")