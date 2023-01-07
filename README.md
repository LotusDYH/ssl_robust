# Self-Supervised Contrastive Learning with Adversarial Perturbations for Robust Pretrained Language Models.

### **Zhao Meng***, **Yihan Dong***, **Mrinmaya Sachan**, **Roger Wattenhofer**   

∗The first two authors contribute equally to this work

[[Paper]](https://aclanthology.org/2022.findings-naacl.8.pdf)

## How to Cite Our Work

```
@inproceedings{sslrobust,
    title = "Self-Supervised Contrastive Learning with Adversarial Perturbations for Defending Word Substitution-based Attacks",
    author = "Meng, Zhao  and
      Dong, Yihan  and
      Sachan, Mrinmaya  and
      Wattenhofer, Roger",
    booktitle = "Findings of the Association for Computational Linguistics: NAACL 2022",
    year = "2022",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-naacl.8",
    doi = "10.18653/v1/2022.findings-naacl.8",
    pages = "87--101",
}
```

## Environment
  - some tips when creating new environment
    - torch cuda version: 10.2 
      - We are not sure whether other versions, e.g. 11.0/11.1, are compatible, so it is better to use version 10.1/10.2
    - torch version: 1.7.1 
    - nvcc version: 10.2 (same as torc cuda version)
    - deepspeed wheel compiled with: torch 1.7 (same as torch version), cuda 10.2 (same as torch cuda version)
    - above information can be seen with command 'ds_report' after installing deepspeed
    - pytorch must be installed before deepspeed
    - gcc/g++: 7.3.0 (>= 5.0)
    - pytorch-lighting: pip install https://github.com/PyTorchLightning/pytorch-lightning/archive/refs/heads/master.zip
  - dataset: bert_attack/dataset
  - fine-tuned model: bert_attack/summary

## Finetune

```
## fine tune: the checkpoint with best val_acc is saved
python fine_tune.py --dataset ag_news --epochs 5 --n_classes 4 --max_length 128 --ngpu 2 --batch_size 32

## re-save the model so it can be read by .from_pretrained direclty (same for adv train and contrastive learning)
python fine_tune.py --dataset ag_news --n_classes 4 --max_length 128 --ngpu 1 --batch_size 32 --checkpoint_path xxx.ckpt --model_path summary/ag_news/model --save_model

## test the accuracy of the model (same for adv train and contrastive learning)
python fine_tune.py --dataset ag_news --n_classes 4 --max_length 128 --ngpu 1 --batch_size 32 --pre_trained_model summary/ag_news/model --test
```

- --dataset: ag_news; imdb; dbpedia
- --n_classed: ag_news: 4; imdb: 2; dbpedia: 14
- --max_length: ag_news: 128; imdb: 512; dbpedia: 128

## Adv Train
```
python adv_train.py --dataset ag_news --n_classes 4 --max_length 128 --ngpu 1 --batch_size 32 --pre_trained_model summary/ag_news/model --summary_dir --synonym cos_sim --cos_sim
```
- different modes to choose candidates:
    - --synonym synonym: use synonyms to choose candidates
    - --synonym none: use BertForMaskedLM to choose candidates
    - --synonym none --cos_sim: use BertForMaskedLM + cos-sim check to choose candidates
    - --synonym cos_sim --cos_sim: use cos-sim matrix to choose candidates (this is the mode we use)
    To use cos-sim matrix, please download counter-fitted-vectors.txt from https://github.com/nmrksic/counter-fitting.
- --n_candidates: max number of candidates for each word (default: 25)
- --max_loops: max loops when attacking (default: 50)
  - n_loops of each sentence = min(max_loops, len(sentence) * 0.5)
- the function of --test flag (--save_model) is as same as the one in fine tune step
- TODO: use (e.g. Universal Sentence Encoder) to check the quality of adv sentences and filter sentences with low quality
  - USE class can be found in utils.py

## TextAttack
```  
python attacks.py --dataset ag_news --pre_trained_model model --n_classes 4 --max_length 128 --attacker TextFoolerJin2019
```  

## Generate Back-Translation Samples in Advance
```  
python trainslation.py --dataset_path dataset/imdb/test_dataset.pkl --output_path dataset/imdb/back_translation_test.pkl
```


## Contrastive Learning (Geometry Attack)
```  
## contrastive learning based on out attack
python contrastive_learning.py --dataset ag_news --n_classes 4 --max_length 128 --ngpu 8 --batch_size 128 --summary_dir summary_cl --synonym none --epochs 30 --max_loops 9

## contrastive learning based on back translation (must generate back-translation samples first)
python contrastive_learning_backtrans.py --dataset ag_news --n_classes 4 --max_length 128 --ngpu 8 --batch_size 128 --summary_dir summary_btcl --epochs 30
```  

## Contrastive Learning (MoCo)
```  
python moco.py --dataset ag_news --n_classes 4 --max_length 128 --ngpu 8 --batch_size 128 --summary_dir summary_moco --synonym none --epochs 15 --max_loops 10
```  