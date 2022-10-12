#This is an anonymous implementation of P-MMF

#The dataset is a processed simulation version for 5% Yelp in dataset/yelp/

#the user-item matrix is trained with the BPR in tmp/bpr_yelp_simulation.npy

#The oracel W_{opt}

python oracle.py

W:6.2344 RRQ: 0.9993 MMF: 0.8810

#P-MMF CPU version

python P-MMF.py --gpu=false

W:6.1984 RRQ: 0.9988 MMF: 0.6031

#P-MMF GPU version

python P-MMF.py --gpu=true

W:6.1927 RRQ: 0.9970 MMF: 0.7086

#Min-reguarlizer

python min-regularizer.py

W:6.1567 RRQ: 0.9984 MMF: 0.2222
