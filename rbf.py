import numpy as np
import math
import sys

# read this : https://stackoverflow.com/questions/23190017/is-pythons-epsilon-value-correct
SUPERSMALL=sys.float_info.epsilon
class inputNode:
    def __init__(self):
        self.inputs=None
        self.outputs=None
    def inputVals(self,ip):
        self.inputs=np.asarray(ip,float)
    def run(self):
        self.outputs=self.inputs    

class rbfNode:
    def __init__(self,CEN,IF,SIG):
        self.outputs=None
        self.center=CEN
        self.ifactor=IF
        self.sig=SIG
    def gaussian_calculation(self,x_v): 
        dist=np.linalg.norm(x_v-self.center)  #self.center for the hidden neuron with the respective ifactor.
        den=self.sig
        return math.exp(-(dist*dist)/(den*den) )
    def run(self,x_v):
        self.outputs=self.gaussian_calculation(x_v)
        return self.outputs
        
        
class outputNode:
    def __init__(self):
        self.inputs=None
        self.outputs=None
        self.weights=None
    def inputVals(self,ip):
        self.inputs=np.asarray(ip,float)
    def run(self):
        self.outputs=np.dot(self.inputs,self.weights)        
