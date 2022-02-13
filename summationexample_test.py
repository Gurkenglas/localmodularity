from math import prod
import torch
import matplotlib.pyplot as plt
import torchviz
import networkgraph
import numpy as np
from tqdm import tqdm

#Set torch to double precision
torch.set_default_dtype(torch.double)

#torch.Tensor.repr = lambda self: self.shape.repr()
torch.Tensor.einsum = lambda self, *args, kwargs: torch.einsum(args[0], self, *args[1:], kwargs)

def halfadder(a,b):
    return a*b, (1-a)*b+(1-b)*a

def adder(ins):
    sum = []
    carry = 0
    activations = []
    for a,b in ins:
        u,v = halfadder(a,b)
        w,x = halfadder(v,carry)
        carry= u+w
        sum.append(x)
        activations.append([a,b,u,v,w,x, carry])
    sum.append(carry)
    return torch.stack(sum)


def duplicate(ins):
    return torch.stack([ins, ins])




def condmutinf(f, shape):
    "Calculate the conditional mutual information between the two groups conditional on the input plus noise"

    #entr_b(a) = b(log(b/a)) = b(log(b/c)) - b(log(a/c))

    #the average of a distribution p is the point relative to which the expected squared distance is minimal
    #the median of a distribution p is the point relative to which the expected distance is minimal
    #the distribution q of a distribution p is the point relative to which the p-expected log(p/q) is minimal

    jacs = []
    color_graph = None
    examplemask = torch.randint(0,2,(1300,))
    for i in range(5):
        

        jacs.append(torch.autograd.functional.jacobian(lambda x: torch.stack(networkgraph.make_dot(examplemask, f(x), show_saved=True)[0]), torch.rand(*shape)).reshape(-1, prod(shape)))
    jac = torch.stack(jacs)
    def entr(m): 
        sings = torch.linalg.svd(jac[:,m>=1,:])[1].log()
        return (sings[:,sings[0]>-10] + (1+np.log(2*np.pi))).sum(1).mean()

    print("jac.shape[1]: " , jac.shape[1])
    mask = torch.randint(0,2,(jac.shape[1],))


    mutinf = float('inf')
    entrss = []
    c = entr(mask+(1-mask))
    n=1
    for i in tqdm(range(50)):
        advantage = 0
        for j in range(mask.shape[0]):
            if advantage==mask[j]:
                mask_new = mask.clone()
                mask_new[j] = 1-mask_new[j]
                

                a, b = entr(mask_new), entr(1-mask_new)
                if a+b-c < mutinf:
                    mutinf = a+b-c
                    mask = mask_new 
                    advantage = 1-advantage
                    exampleinput = torch.rand(*shape, requires_grad=True)
                    networkgraph.make_dot(mask, f(exampleinput), show_saved=True)[1].render(f'graph_{n}', format='jpg')
                    n+=1
                    print(a, b, a+b-c)
        advantage = 1-advantage
    
    #assemble graphs into a video
    import os
    os.system('ffmpeg -r 10 -i graph_%d.jpg -vcodec mpeg4 -y graph.mp4')
    os.system('rm graph_*.jpg')

    print("mutinf: " , mutinf.item())
condmutinf(adder, (13,2))

#

