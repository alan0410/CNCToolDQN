import os
os.environ.setdefault("TF_NUM_THREADS", "1")
import numpy as np
import matplotlib.pyplot as plt
import collections
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import seaborn as sns
import math
import time
import tqdm
import argparse
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler

from scipy.stats import skew

from model import *
from Enviroment import *
from scoring_function import scoring_function

anomaly_index_final = np.load('Data/model_inputs/anomaly_index_final.npy')
unknown_train_index = np.load('Data/model_inputs/unknown_train_index.npy')
spindleload_train_data = np.load('Data/model_inputs/spindleload_train_data.npy')
servox_train_data = np.load('Data/model_inputs/servox_train_data.npy')
servoy_train_data = np.load('Data/model_inputs/servoy_train_data.npy')
servoz_train_data = np.load('Data/model_inputs/servoz_train_data.npy')
life_train_data = np.load('Data/model_inputs/life_train_data.npy')
anomaly_train_data = np.load('Data/model_inputs/anomaly_train_data.npy')

spindleload_test_data = np.load('Data/model_inputs/spindleload_test_data.npy')
servox_test_data = np.load('Data/model_inputs/servox_test_data.npy')
servoy_test_data = np.load('Data/model_inputs/servoy_test_data.npy')
servoz_test_data = np.load('Data/model_inputs/servoz_test_data.npy')
life_test_data = np.load('Data/model_inputs/life_test_data.npy')


train_label = np.zeros(len(life_train_data))
train_label[anomaly_index_final] = 1
train_label = train_label.astype(int)

normal_index = np.where(np.array(anomaly_train_data) == 'Normal')[0]
unknown_index = np.where(np.array(anomaly_train_data) == 'Unknown')[0]

def scoring_train_data(model):
    
    train_length = spindleload_train_data.shape[0]
    mok = train_length //100 
    nameoji = train_length %100 
    
    train_result_list = []

    for i in range(mok+1):
        spindleload_train_data_batch = spindleload_train_data[(i*100):(100*i+100)]
        servox_train_data_batch = servox_train_data[(i*100):(100*i+100)]
        servoy_train_data_batch= servoy_train_data[(i*100):(100*i+100)]
        servoz_train_data_batch= servoz_train_data[(i*100):(100*i+100)]
        life_train_data_batch = life_train_data[(i*100):(100*i+100)]
        result = model.forward(spindleload_train_data_batch,servox_train_data_batch, servoy_train_data_batch,  servoz_train_data_batch, life_train_data_batch )[1].squeeze(1)[:,1]

        if i == (mok):
            spindleload_train_data_batch = spindleload_train_data[(i*100):(100*i+nameoji+1)]
            servox_train_data_batch = servox_train_data[(i*100):(100*i+nameoji+1)]
            servoy_train_data_batch= servoy_train_data[(i*100):(100*i+nameoji+1)]
            servoz_train_data_batch= servoz_train_data[(i*100):(100*i+nameoji+1)]
            life_train_data_batch = life_train_data[(i*100):(100*i+nameoji+1)]
            result = model.forward(spindleload_train_data_batch,servox_train_data_batch,servoy_train_data_batch, servoz_train_data_batch, life_train_data_batch )[1].squeeze(1)[:,1]

        train_result_list.append(result.tolist())
    
    result_list_flat_train = np.array([item for sublist in train_result_list for item in sublist])
    #print(result_list_flat_train.shape)
    
    sns.distplot(result_list_flat_train)
    
    threshold_list = np.arange(0.5, 0.95, 0.25)
    f1_score_list = []
    
    for threshold in threshold_list:
        result_list_flat_train = np.where(np.array(result_list_flat_train) > , 1, 0)
        f1score = f1_score(train_label, result_list_flat_train, average= 'macro')
        #onfusion = confusion_matrix(train_label, result_list_flat_train)
        f1_score_list.append(f1_score)
    
    best_thrshold = threshold_list[np.argmax(f1_score_list)]
    print("The best thershold: ", best_threshold )
    print("f1_score: ", np.max(f1_score_list))
    
    return best_thrshold 

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type = str, default = 'C:/')
    parser.add_argument('--dir_savefigure', type = str, default = 'C:/')
    parser.add_argument("--gpu", type=str, default='0')
    parser.add_argument("--maxlen", type=int, default=300)
    parser.add_argument("--L", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.005)
    parser.add_argument("--tau", type=float, default=0.1)
    
    args = parser.parse_args()
    
    SAVE_DIRECTORY = args.dir
    DATA_SOURCE_PATH = SAVE_DIRECTORY + '/model_inputs/'
    if 'Model' not in os.listdir(SAVE_DIRECTORY):
        os.mkdir(SAVE_DIRECTORY + '/Model')
    MODEL_STORAGE_PATH = SAVE_DIRECTORY + '/Model'
    GPU_NUM = args.gpu
    max_len = args.maxlen
    

    device = torch.device(
        f"cuda:{GPU_NUM}") if torch.cuda.is_available() else torch.device("cpu")
    
    torch.set_num_threads(4)

    score_list = []
    env = My_policy(max_len)
    q= Model(device,max_len)#.to(device)
    q_target = Model(device,max_len)#.to(device)
    q_target.load_state_dict(q.state_dict()) 
    memory = ReplayBuffer(device)
    print_interval = 10
    score = 0.0

    for n_epi in range(101):
        torch.set_num_threads(4)

        start = time.time()
        epsilon = max(0.01, 0.98 - 0.02 * (n_epi))
        sp_u, x_u, y_u, z_u, life_u, sp_a, x_a, y_a, z_a, life_a  = env.reset()
        done = False

        # Initialization       
        
        #unknown_index
        
        now_index = np.random.randint(len([unknown_index]))
        
        s1 = spindleload_train_data[unknown_train_index][now_index]
        s2 = servox_train_data[unknown_train_index][now_index]
        s3 = servoy_train_data[unknown_train_index][now_index]
        s4 = servoz_train_data[unknown_train_index][now_index]
        #s5 = np.array(train_json['Speed'])[unknown_index][now_index]
        
        s_life = [life_train_data[unknown_train_index][now_index]]
        
        s1, s2, s3, s4, = torch.tensor([s1]), torch.tensor([s2]), torch.tensor([s3]),  torch.tensor([s4])

        anomaly_label = anomaly_train_data[unknown_train_index][now_index]
        #pid =  np.array(train_json['seqlen'])[unknown_index][now_index]
        step = 0
        r_list = []
        r_extrinsic_list, r_intrinsic_list = [], []
        
        while not done:
            a, action_prob = q.sample_action(s1,s2,s3, s4, s_life, epsilon)    
            
            x_rep, out = q.forward(s1,s2, s3,s4,  s_life)

            s = x_rep

            sp_prime, x_prime, y_prime,  z_prime,  s_prime, life_prime, r, r_extrinsic, r_intrinsic, done, anomaly_label_shat= env.step(q, a, action_prob, s, s1, s2, s3, s4,  s_life, anomaly_label)

            done_mask = 0.0 if done else 1.0

            r_list.append(round(r,2))
            memory.put((s[0].tolist(), s1[0].tolist(), s2[0].tolist(), s3[0].tolist(),s4[0].tolist(), s_life, a, r/5 , s_prime.tolist() , sp_prime.tolist(), x_prime.tolist(), y_prime.tolist(),  z_prime.tolist(), life_prime, done_mask, anomaly_label, anomaly_label_shat  ))

            s = s_prime
            s1 = torch.tensor([sp_prime])
            s2 = torch.tensor([x_prime])
            s3 = torch.tensor([y_prime])
            s4 = torch.tensor([z_prime])
            s_life = life_prime
            score += r
            anomaly_label = anomaly_label_shat
            #pid = pid_hat
            
            step += 1

            if done :

                score_list.append(score/step)
                step = 0
                print("n_episode: {}, score : {:.1f}, n_buffer {}, eps : {:.1f}% | - 2 ratio {:.1f}% | -1 ratio {:.1f}%".format(n_epi,
                                                    score, memory.size(), epsilon * 100,  r_list.count(-2)/len(r_list)*100, r_list.count(-1)/len(r_list)*100  ))
                
                score = 0
                
                if n_epi == 0:
                    print("time spent to perform an episode :", time.time() - start)
                if n_epi % 5 ==0:
                    print("r list: ", r_list)
                else:
                    pass
                break


        if n_epi % print_interval == 0 and n_epi != 0:
            q_target.load_state_dict(q.state_dict())
        if n_epi % 5 == 0 and n_epi != 0:    
            best_threshold = scoring_train_data(q_target)

    torch.save(q_target.state_dict(), MODEL_STORAGE_PATH)  
    model = Model(device, max_len)
    model.load_state_dict(torch.load(MODEL_STORAGE_PATH))

    train_length = spindleload_train_data.shape[0]
    mok = train_length //100 
    nameoji = train_length %100
    train_result_list = []
    
    #batch 에 따라 모델 결과 산출
    for i in tqdm(range(mok+1)):
        spindleload_train_data_batch = spindleload_train_data[(i*100):(100*i+100)]
        servox_train_data_batch = servox_train_data[(i*100):(100*i+100)]
        servoy_train_data_batch= servoy_train_data[(i*100):(100*i+100)]
        servoz_train_data_batch= servoz_train_data[(i*100):(100*i+100)]
        life_train_data_batch = life_train_data[(i*100):(100*i+100)]
        result = model.forward(spindleload_train_data_batch,servox_train_data_batch, servoy_train_data_batch,  servoz_train_data_batch, life_train_data_batch )[1].squeeze(1)[:,0]
        
        if i == (mok):
            spindleload_train_data_batch = spindleload_train_data[(i*100):(100*i+nameoji+1)]
            servox_train_data_batch = servox_train_data[(i*100):(100*i+nameoji+1)]
            servoy_train_data_batch= servoy_train_data[(i*100):(100*i+nameoji+1)]
            servoz_train_data_batch= servoz_train_data[(i*100):(100*i+nameoji+1)]
            life_train_data_batch = life_train_data[(i*100):(100*i+nameoji+1)]
            result = model.forward(spindleload_train_data_batch,servox_train_data_batch,servoy_train_data_batch, servoz_train_data_batch, life_train_data_batch )[1].squeeze(1)[:,0]
        
        train_result_list.append(result.tolist()) 

    result_list_flat_train = np.array([item for sublist in train_result_list for item in sublist]).reshape(-1, 1) 


    test_result_list = []

    #batch 에 따라 모델 결과 산출
    for i in tqdm(range(mok+1)):
        spindleload_test_data_batch = spindleload_test_data[(i*100):(100*i+100)]
        servox_test_data_batch = servox_test_data[(i*100):(100*i+100)]
        servoy_test_data_batch= servoy_test_data[(i*100):(100*i+100)]
        servoz_test_data_batch= servoz_test_data[(i*100):(100*i+100)]
        
        life_test_data_batch = life_test_data[(i*100):(100*i+100)]
        result = model.forward(spindleload_test_data_batch,servox_test_data_batch, servoy_test_data_batch, servoz_test_data_batch,  life_test_data_batch )[1].squeeze(1)[:,1]
        
        if i == (mok):
            spindleload_test_data_batch = spindleload_test_data[(i*100):(100*i+nameoji+1)]
            servox_test_data_batch = servox_test_data[(i*100):(100*i+nameoji+1)]
            servoy_test_data_batch= servoy_test_data[(i*100):(100*i+nameoji+1)]
            servoz_test_data_batch= servoz_test_data[(i*100):(100*i+nameoji+1)]
            life_test_data_batch = life_test_data[(i*100):(100*i+nameoji+1)]
            result = model.forward(spindleload_test_data_batch,servox_test_data_batch, servoy_test_data_batch, servoz_test_data_batch,  life_test_data_batch )[1].squeeze(1)[:,1]

        test_result_list.append(result.tolist())
        
    result_list_flat_test = np.array([item for sublist in test_result_list for item in sublist]).reshape(-1, 1) 
    
    plt.rc('font', size=20)        # font
    plt.rc('axes', labelsize=25)   # x,y axis label font
    plt.rc('xtick', labelsize=20)  # xtick font
    plt.rc('ytick', labelsize=20)  # ytick font
    plt.rc('legend', fontsize=20)  # Legend font
    plt.rc('figure', titlesize=75) 

    scaler = MinMaxScaler( feature_range=(0, 1))
    scaler.fit(np.array(result_list_flat_train))

    result_list_flat_train = scaler.transform(result_list_flat_train)
    result_list_flat_train = 1- result_list_flat_train
    life_train_data = 1- life_train_data
    
    # Q(s, a0) plot (Training set)
    plt.figure(figsize= (20,10))
    plt.scatter(np.arange(len(result_list_flat_train )),  np.array(result_list_flat_train), c= 'r', label = 'Prediction',linewidth=2, s= 1.5)
    plt.plot(np.arange(len(life_train_data )), np.array(life_train_data ), c= 'b', label = 'Life', linewidth=3)
    #plt.legend()
    plt.title("CNCToolDQN Q(s, a0 ) - OP102A End mill (Training set)")
    plt.ylim(0, 1)
    plt.xlim(0, len(life_train_data ))
    plt.xlabel("Time (cycles)")
    plt.grid()
    plt.ylabel("Prediction")
    plt.savefig(args.dir_savefigure+ '/Q(s,a0).png')
    plt.show()
    
    RUL_list_train, mean_life_list_train ,mean_list, baseline_eliminated,  score_list = scoring_function (result_list_flat_train, life_train_data,  np.array(train_json['batch_number'])[idx_train], args.L, args.alpha, args.tau)#L, alpha , tau
    
    # RUL plot (Training set)
    plt.figure(figsize= (20,10))
    plt.plot(np.arange(len(RUL_list_train)),RUL_list_train, c= 'r', label = 'Prediction', linewidth = 3)
    plt.plot(np.arange(len(mean_life_list_train )), np.array(mean_life_list_train ) , c= 'b', label = 'Life', linewidth=3)
    plt.grid()
    plt.title("TH score - OP102A End Mill(Training set)")
    plt.xlabel("Time (batch)")
    plt.ylim(0, 1)
    plt.xlim(0, )
    plt.ylabel("TH score")
    plt.savefig(args.dir_savefigure + '/TH_score.png')
    plt.show()
    
    
    result_list_flat_test = scaler.transform(result_list_flat_test)
    result_list_flat_test = 1- result_list_flat_test
    life_test_data = 1- life_test_data

    # Q(s, a0) plot (Test set)
    plt.figure(figsize= (20,10))
    plt.scatter(np.arange(len(result_list_flat_test )),  np.array(result_list_flat_test), c= 'r', label = 'Prediction',linewidth=2, s= 1.5)
    plt.plot(np.arange(len(life_test_data )), np.array(life_test_data ), c= 'b', label = 'Life', linewidth=3)
    #plt.legend()
    plt.title("CNCToolDQN Q(s, a0 ) - OP102A End mill (Test set)")
    plt.ylim(0, 1)
    plt.xlim(0, len(life_test_data ))
    plt.xlabel("Time (cycles)")
    plt.grid()
    plt.ylabel("Prediction")
    plt.savefig(args.dir_savefigure+ '/Q(s,a0)_test.png')
    plt.show()
    
    RUL_list_test, mean_life_list_test ,mean_list, baseline_eliminated,  score_list = scoring_function (result_list_flat_test, life_test_data,  np.array(test_json['batch_number'])[idx_test], args.L, args.alpha, args.tau)#L, alpha , tau
    
    # RUL plot (test set)
    plt.figure(figsize= (20,10))
    plt.plot(np.arange(len(RUL_list_test)),RUL_list_test, c= 'r', label = 'Prediction', linewidth = 3)
    plt.plot(np.arange(len(mean_life_list_test )), np.array(mean_life_list_test ) , c= 'b', label = 'Life', linewidth=3)
    plt.grid()
    plt.title("TH score - OP102A End Mill(Test set)")
    plt.xlabel("Time (batch)")
    plt.ylim(0, 1)
    plt.xlim(0, )
    plt.ylabel("TH score")
    plt.savefig(args.dir_savefigure + '/TH_score_test.png')
    plt.show()
    
    
    #Printing the f1 score of test_sset
    
    from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
    from sklearn import metrics

    r = np.where(result > best_threshold, 1 , 0)

    accuracy = round(accuracy_score(l, r),3)
    f1_score = round(metrics.f1_score(l, r, average ='macro'), 3)

    print("{} Accuracy:{} F1 score:{} ".format(round(i,2), accuracy, f1_score))
    print("confusion:\n", confusion_matrix(l,r))
