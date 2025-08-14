---
license: mit
datasets:
- clip-rt/modified_libero_hdf5
language:
- en
tags:
- robotics
- vla
- clip
- contrastive_learning
---

# CLIP-RT Finetuned on LIBERO-Goal

We finetune the original [CLIP-RT model](https://clip-rt.github.io/) with a 300M-parameter action decoder to enable continuous action prediction. This checkpoint is the model finetuned on [LIBERO](https://libero-project.github.io/main.html) goal task suite.


## Hyperparemeters

| Category             | Details                                                             |
|----------------------|---------------------------------------------------------------------|
| **Train**            | 8 × H100 GPUs, each with 80GB VRAM (batch size: 256)                |
| **Model size**       | 1.3B (CLIP-RT base + 0.3B action decoder)                           |
| **Action dimension** | 7D end-effector action × 8 action chunks                            |
| **Loss**             | L1 regression                                                       |
| **Epochs**           | 128                                                                 |
| **Performance**      | 92.2% success rate on the LIBERO-Goal task suite                    |
| **Throughput**       | 163Hz                                                               |
| **Inference**        | One GPU with 9GB VRAM                                               |

## Usage Instructions
If you want to evaluate this model on the LIBERO simulator, please refer to the [clip-rt github repository](https://github.com/clip-rt/clip-rt/tree/main/libero). 

## Citation

```bibtex
@article{kang2024cliprt,
  title={CLIP-RT: Learning Language-Conditioned Robotic Policies from Natural Language Supervision},
  author={Kang, Gi-Cheon and Kim, Junghyun and Shim, Kyuhwan and Lee, Jun Ki and Zhang, Byoung-Tak},
  journal={arXiv preprint arXiv:2411.00508},
  year = {2024}
}
```