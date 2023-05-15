## __Link collection text-to-image generation AIs__
### List of websites/providers of image generation powered by AI
1. [Fotor](https://www.fotor.com/images/create)
2. [Craiyon](https://www.craiyon.com/)
3. [pixray](https://replicate.com/pixray/text2image/versions/5c347a4bfa1d4523a58ae614c2194e15f2ae682b57e3797a5bb468920aa70ebf)
4. [Adobe Firefly (beta)](https://firefly.adobe.com/generate/images)
5. [VQGAN](https://ljvmiranda921.github.io/notebook/2021/08/08/clip-vqgan/) + [CLIP](https://openai.com/research/clip) on Google Colab


### <ins> Fotor, Craiyon, pixray and AdobeFirefly </ins>
For using these image generation websites one only needs to precisely describe a scene which should be generated. Therefore keywords like e.g. "spray", "fog", "wet street surface" can be used. Some websites offer a gew parameters for configuration as well to influence the outcome.


### <ins> [VQGAN](https://ljvmiranda921.github.io/notebook/2021/08/08/clip-vqgan/) + [CLIP](https://openai.com/research/clip) on Google Colab </ins>
There is a given [setup](https://colab.research.google.com/github/justinjohn0306/VQGAN-CLIP/blob/main/VQGAN%2BCLIP(Updated).ipynb#scrollTo=g7EDme5RYCrt&uniqifier=1) available on Google Colab, which can be filled with "init image" and "target images" (more than one possible) and some further parameters. A detailed description can be found [here](https://docs.google.com/document/d/1Lu7XPRKlNhBQjcKr8k8qRzUzbBW7kzxb5Vu72GMRn2E/edit#heading=h.7bt2ltvefvkz).
In the subfolder "03_VQGAN_CLIP" results of different parametrizations can be found.


## __Link collection text-to-image inpainting AIs__
### List of websites/providers of image inpaiting powered by AI
1. [Hugging Face](https://huggingface.co/spaces/multimodalart/stable-diffusion-inpainting)
2. [getimg.ai](https://getimg.ai/text-to-image)
3. [NVIDIA](https://www.nvidia.com/research/inpainting) ([not working](https://github.com/NVIDIA/partialconv/issues/24) [15.05.2023])
4. [Adobe Firefly (beta)](https://firefly.adobe.com/generate/images) (currently not available [15.05.2023])


### <ins> Hugging Face and getimg.ai </ins>
For using inpainting which is AI-based one needs to upload an image which should be changed. Then configure the AI's parameters to influence how strong the image is allowed to be changed (for getimg.ai e.g. "image guidance" as parameter).
Based on the text prompt input the image is then changed. Further information on the configuration can be found [here for getimg.ais](https://getimg.ai/guides) and here for [Hugging Face](https://huggingface.co/docs).


## __Conlusion so far__
- Most of the text-to-image generators are more likely used for artistic generation of images instead of using them for realistic image augmentation for weather conditions.
- Adding an animal like a dog for example into an image is not a problem, but adding spray which is not a separable object is quite challenging.
- 3D dependencies are not correctly interpreted by the text interpretation part of the text-to-image AIs.
- The keyword spray is not well known by the trained text interpretation networks.
- Especially the correct placement and visualisation of spray is in nearly all cases physically not correct.