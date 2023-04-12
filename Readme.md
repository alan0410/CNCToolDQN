CNCToolDQN implementation and TH score estimation (First uploaded in April/12/2023)

Data sample will be uploaded soon.

How to run:

python main.py --dir='C:/' --dir_savefigure= 'C:/'  --gpu='0' --maxlen=300 --L= 1 --alpha=0.005 --tau=1


1) --dir: Data directory. ex) "C:/Users/Desktop/Data/"
2) --dir_savefigure : The directory to save TH score plot. ex) "C:/Users/Desktop/figure/"
3) --gpu: GPU index to use (default: '0')
4) --maxlen: Window size
5) --L: Lower limit to calculate p_l, which is the probability of Z score lower than -L (default = 1)
6) --alpha: An hyperparameter to estimate TH score (default = 0.005) 
7) --tau: An hyperparameter to estimate TH score (default = 0.1)
