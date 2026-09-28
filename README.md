# cringe-neuro-translate
Still work-in-progress

elevenlabs-alike purposely very poor translation EN --> RU pipeline. All models are working locally and free for personal use.

## List of used models:
1. [Openai Whisper](https://github.com/openai/whisper) - Transcription
2. [nvidia/diar_sortformer_4spk-v1](https://huggingface.co/nvidia/diar_sortformer_4spk-v1) - Diarization
3. [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) - Translation/Summarization
4. [Chatterbox-multilingual v3](https://github.com/resemble-ai/chatterbox) - text-to-speech

> [!NOTE]
> Most of the code (~95%) written by LLM such as ChatGPT


## Hardware

Nvidia GeForce RTX 3080 12gb

```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 576.80                 Driver Version: 576.80         CUDA Version: 12.9     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3080      WDDM  |   00000000:07:00.0  On |                  N/A |
| 30%   37C    P8             29W /  280W |    1392MiB /  12288MiB |     20%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
```
