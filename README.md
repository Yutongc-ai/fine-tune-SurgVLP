This repository is based on:

[1] [Learning Multi-modal Representations by Watching Hundreds of Surgical Video Lectures](https://arxiv.org/abs/2307.15220)          
[2] [HecVL: Hierarchical Video-Language Pretraining for Zero-shot Surgical Phase Recognition](https://arxiv.org/abs/2405.10075)       
Presented at MICCAI 2024           
[3] [Procedure-Aware Surgical Video-language Pretraining with Hierarchical Knowledge Augmentation](https://arxiv.org/abs/2410.00263)           
Presented at NeurIPS 2024   

### How to run
```
python few_shot.py --method <method_name>
```

Available `method_name` can be found at `tests/config_surgvlp_few_shot.py` and configs can be set at this file also. 