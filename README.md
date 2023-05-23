# CAR: Conceptualization-Augmented Reasoner for Zero-Shot Commonsense Question Answering

This is the official code and data repository for the paper: CAR: Conceptualization-Augmented Reasoner for Zero-Shot
Commonsense Question Answering.

![CAR](./demo/overview.png "CAR Framework Overview")

## 1. Download Data & Model Checkpoint

All conceptualization data, including the discriminator and generator models, can be downloaded
at [this link](https://hkustconnect-my.sharepoint.com/:f:/g/personal/wwangbw_connect_ust_hk/EnA7X6PkeE5Dll9sdlwxuG4BH8zw-Bpdtc5kw3L70Shu5g).
Please refer to our [previous ACL2023 paper](https://arxiv.org/abs/2305.04808) for more details.

The training data and model checkpoint of our best DeBERTa-v3-Large QA model trained by CAR can be downloaded
at [this link](https://hkustconnect-my.sharepoint.com/:f:/g/personal/wwangbw_connect_ust_hk/EqC6BjWPGi1IgEPLbfMWk7gBFAn-cpfQAmTUsQBoUzVkqw).

## 2. Required Packages

Required packages are listed in `requirements.txt`. Install them by running:

```bash
pip install -r requirements.txt
```

## 3. Model Training

## 4. Citing this Work

Please use the bibtex below for citing our paper:

```bibtex
@inproceedings{CAR,
  author       = {Weiqi Wang and
                  Tianqing Fang and
                  Wenxuan Ding and
                  Baixuan Xu and
                  Xin Liu and
                  Yangqiu Song and 
                  Antoine Bosselut},
  title        = {CAR: Conceptualization-Augmented Reasoner for Zero-Shot Commonsense Question Answering},
  year         = {2023}
}
```

## 5. Acknowledgement

The authors of this paper are supported by the NSFC Fund (U20B2053) from the NSFC of China, the RIF (R6020-19 and
R6021-20), and the GRF (16211520 and 16205322) from RGC of Hong Kong, the MHKJFS (MHP/001/19) from ITC of Hong Kong and
the National Key R&D Program of China (2019YFE0198200) with special thanks to HKMAAC and CUSBLT.
We also thank the UGC Research Matching Grants (RMGS20EG01-D, RMGS20CR11, RMGS20CR12, RMGS20EG19, RMGS20EG21,
RMGS23CR05, RMGS23EG08).