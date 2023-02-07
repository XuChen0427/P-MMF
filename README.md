# P-MMF: Provider Max-min Fairness Re-ranking in Recommender System of WWW'23
## Xu Chen, Ph.D. student of Renming University of China, GSAI
Any question, please mail to xc_chen@ruc.edu.cn

#The dataset is a processed simulation version for 5% Yelp in dataset/yelp/

#the user-item matrix is trained with the BPR in tmp/bpr_yelp_simulation.npy

#For the offline program oracel W_{opt}, run

```bash
python oracle.py
```

W:6.2344 RRQ: 0.9993 MMF: 0.8810

#For the P-MMF CPU version, run

'''bash
python P-MMF.py --gpu=false
'''

W:6.1984 RRQ: 0.9988 MMF: 0.6031

#For the P-MMF GPU version
```bash
python P-MMF.py --gpu=true
```
W:6.1927 RRQ: 0.9970 MMF: 0.7086

#For the Min-reguarlizer algorithm, run

```bash
python min-regularizer.py
```

W:6.1567 RRQ: 0.9984 MMF: 0.2222


##For citation, please cite the following bib
```
@inproceedings{Xu-PMMF-WWW23,
author = {Xu, Chen and Chen, Sirui and Xu, Jun and Shen, Weiran and Zhang, Xiao and Wang, Gang and Dong, Zhenghua},
title = {P-MMF: Provider Max-min Fairness Re-ranking in Recommender System},
year = {2023},
isbn = {978-1-4503-9416-1/23/04},
publisher = {Association for Computing Machinery},
address = {Austin, TX, USA},
doi = {10.1145/3543507.3583296},
booktitle = {{Proceedings of the ACM Web Conference 2023 (WWW '23)},
series = {WWW '22}
}
```
