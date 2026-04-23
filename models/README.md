# Models

This folder is the local Hugging Face cache and model store. It is intentionally ignored by Git because it contains large model weights, tokenizer files, and cached benchmark datasets.

Expected cache contents include:

- `hub/`: Hugging Face model and dataset snapshots.
- `datasets/`: cached datasets for `lm_eval`.
- auth/cache helper files such as `token` and `stored_tokens`.

Scripts mount this directory into Docker as `/root/.cache/huggingface` and set offline mode:

```bash
-e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
-v /home/saketh-msc/quantization/models:/root/.cache/huggingface
```

Keep this README tracked, but do not commit the model cache itself.
