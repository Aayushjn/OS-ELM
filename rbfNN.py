from rbf import *
from pandas import read_csv
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import math
import statistics
import matplotlib.pyplot as plt


def beta_pseudoinverse(h_matrix,t_matrix):
    h_transpose=h_matrix.transpose()
    mul=np.matmul(h_transpose,h_matrix)
    pseudoinv=np.linalg.pinv(mul)
    beta=np.matmul(pseudoinv,h_transpose)
    temp=t_matrix
    for row in temp:
        row.transpose()
    beta=np.matmul(beta,temp)
    return beta,pseudoinv

def clusters_sigmas(attri_array,hidden_nodes): #Chunk_i's set of input attributes - np array.
    kmeans = KMeans(n_clusters=hidden_nodes) 
    kmeans.fit(attri_array) 
    centers = kmeans.cluster_centers_
    d=0
    for center1 in centers:
        for center2 in centers:
            dist=np.linalg.norm(center1-center2)
            if d<dist: 
                d=dist
    sig=d/math.sqrt(2*hidden_nodes)
    return sig,centers

def next_power_of_2(x):     #return next power of two
    if x == 0:
        return 1 
    else:
        return 2**math.ceil(math.log2(x)) 



def main():
    
    #Read dataset 
    data = read_csv('./datasets/Abalone.csv')
    data[data.keys()[-1]] = LabelEncoder().fit_transform(data[data.keys()[-1]].astype('str'))
    x = data.iloc[:, :data.shape[1] - 1]
    t = data.iloc[:, data.shape[1] - 1]
    
    ## GENERALIZE THIS ADITI!!!!
    #for val in x:
    #    x[val] /= 29
        
    
    x_train, x_test, t_train, t_test = train_test_split(x, t, train_size=0.8, test_size=0.2)

    #Calculate number of input, hd, output nodes

    n_input=len(data.keys())-1
    n_output=len(set(data[data.keys()[-1]]))
    #n_hidden=next_power_of_2((n_input+n_output)/2)*2
    n_hidden=20
    # Divide the dataset into two parts:-
    #   (1) for the initial training phase
    #   (2) for the sequential training phase
    # NOTE: The number of training samples for the initial training phase
    # must be much greater than the number of the model's hidden nodes.
    # Here we assign int(10 * n_hidden_nodes) training samples
    # for the initial training phase.

    border = int(10 * n_hidden)
    x_train_init = x_train.values[:border]
    x_train_seq = x_train.values[border:]
    t_train_init = t_train.values[:border]
    t_train_seq = t_train.values[border:]


    t_temp=[[0 for i in range(len(data.keys()[-1]))] for j in range(len(t_train_init))]
    for i in range(len(t_train_init)):
        t_temp[i][t_train_init[i]]=1
    t_train_init=t_temp
    t_temp=[[0 for i in range(len(data.keys()[-1]))] for j in range(len(t_train_seq))]
    for i in range(len(t_train_seq)):
        t_temp[i][t_train_seq[i]]=1
    t_train_seq=t_temp

    print(t_train_seq)
    #construct the network 
    ifactors=[np.random.random(1)[0] for i in range(0,n_hidden)]
    sigma,centers=clusters_sigmas(x_train_init,n_hidden)

    hidden_nodes=[rbfNode(centers[i],ifactors[i],sigma) for i in range(0,n_hidden)]
    
    #initial learning phase
    h_matrix=[]
    for x in x_train_init:
        h_row=[]
        for node in hidden_nodes:
            h_row.append(node.run(x))
        h_matrix.append(h_row)  
    h_matrix=np.asmatrix(h_matrix)
    beta_matrix,p_matrix=beta_pseudoinverse(h_matrix,np.asmatrix(t_train_init))

    #splitting the dataset 
    m=math.ceil((len(x_train_seq)//20))
    beta_prev=beta_matrix
    p_prev=p_matrix
    ### SEQUENTIAL LEARNING PHASE BEGINS HERE

    for k in range(0,m):
        x_k_seq=x_train_seq[k*20:(k+1)*20]
        t_k_seq=t_train_seq[k*20:(k+1)*20]
        h_k_seq=[]
        for x in x_k_seq:
            h_k_row=[]
            for node in hidden_nodes:
                h_k_row.append(node.run(x))
            h_k_seq.append(h_k_row)
        h_k_seq=np.asmatrix(h_k_seq,float)
        
        #Some long matrix calculations
        mulCalc=np.matmul(p_prev,h_k_seq.transpose())
        tempCalc=np.matmul(h_k_seq,p_prev)
        tempCalc=np.matmul(tempCalc,h_k_seq.transpose())
        tempCalc=np.identity(tempCalc.shape[1])+tempCalc
        tempCalc=np.linalg.pinv(tempCalc)
        mulCalc=np.matmul(mulCalc,tempCalc)
        mulCalc=np.matmul(mulCalc,h_k_seq)
        mulCalc=np.matmul(mulCalc,p_prev)

        #And it comes to use here
        p_k_seq=p_prev-mulCalc
        
        #Some more long matrix calculations
        mulCalc2=np.matmul(p_k_seq,h_k_seq.transpose())
        tempCalc=np.matmul(h_k_seq,beta_prev)
        tempCalc=np.matrix(t_k_seq)-tempCalc
        mulCalc2=np.matmul(mulCalc2,tempCalc)

        #And it comes to use here
        beta_k_seq=beta_prev+mulCalc2
        p_prev=p_k_seq
        beta_prev=beta_k_seq
    

    #print(beta_prev.shape)
    ### SEQUENTIAL LEARNING PHASE ENDS
    # now we have final weights stored in beta_prev
    # we use these weights in the output layer node to run the code and calculate the accuracy on the test data        

    ## TESTING PHASE BEGINS
    x_test=x_test.values[:]
    t_test=t_test.values[:]
    t_temp=[[0 for i in range(len(data.keys()[-1]))] for j in range(len(t_test))]
    for i in range(len(t_test)):
        t_temp[i][t_test[i]]=1
    t_train_init=t_test
    h_test_seq=[]
    for x in x_test:
        h_test_row=[]
        for node in hidden_nodes:
            h_test_row.append(node.run(x))
        h_test_seq.append(h_test_row)
    h_test_seq=np.asmatrix(h_test_seq)

    results=np.matmul(h_test_seq,beta_prev)
    results=results.tolist()
    correct=0
    for i  in range(len(results)):
        r=results[i]
        if r.index(max(r))==t[i]:
            correct+=1
    print(correct/len(results))
        
    # plt.subplot(2, 1, 1)
    # plt.plot(t_test[:40],'bo')
    # plt.subplot(2,1,2)
    # plt.plot(results[:40],'ro')
    # plt.show()
if __name__=="__main__":
    main()
