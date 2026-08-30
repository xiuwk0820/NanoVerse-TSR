# Classifier init weights

Only `bpe_simple_vocab_16e6.txt.gz` is tracked. Download CLIP checkpoints separately.

| Local name | Official file |
|------------|----------------|
| `ViT-B-16-OpenAI.pth` | [OpenAI ViT-B/16](https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt) |
| `ViT-B-32-OpenAI.pth` | [OpenAI ViT-B/32](https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt) |

```bash
bash scripts/download_init_weights.sh
```

See [docs/WEIGHTS.md](../../docs/WEIGHTS.md).
