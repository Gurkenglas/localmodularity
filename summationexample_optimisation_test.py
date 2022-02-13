import functools
import itertools
from math import prod
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import torchviz
import networkgraph
import numpy as np
from tqdm import tqdm
from scipy.cluster import hierarchy

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
    for i in tqdm(range(5)):
        jacs.append(torch.autograd.functional.jacobian(lambda x: torch.stack(networkgraph.make_dot(examplemask, f(x), show_saved=True)[0]), torch.rand(*shape)).reshape(-1, prod(shape)))
    jac = torch.stack(jacs)
    print("jac.shape[1]: " , jac.shape[1])

    count = 0   
    def cluster(t):
        nonlocal count
        t.index = count
        count += 1
        return t

    cov = jac @ jac.T

    @functools.cache
    def mutinf(s):
        def entr(m): 
            sings = torch.linalg.svd(jac[:,m>=1,:])[1].log()
            return (sings[:,sings[0]>-10] + (1+np.log(2*np.pi))).sum(1).mean()
        a,b  = sorted(list(s), key = lambda t: t.index)
        return (entr(a) + entr(b) - entr(a+b))/(a+b).count_nonzero()

    clusters = set([cluster(t) for t in torch.eye(jac.shape[1]).unbind()])
    linkage = []        
    while len(clusters)>1:
        a,b = max(itertools.combinations(clusters, 2), key=mutinf)
        clusters.remove(a)
        clusters.remove(b)
        c = cluster(a+b)
        clusters.add(c)

        linkage.append([a.index, b.index, c.count_nonzero().log().item(), c.count_nonzero().item()])

    plt.figure()    
    dn = hierarchy.dendrogram(linkage)
    plt.savefig("dendrogram.jpg")
    plt.show()

    leaflist = dn["leaves"]
    colorlist = dn["leaves_color_list"]
    oranges_colorlist = [mpl.colors.to_hex(colorlist[leaflist.index(i)]) for i in range(len(leaflist))]

    exampleinput = torch.rand(*shape, requires_grad=True)
    networkgraph.make_dot(oranges_colorlist, f(exampleinput), show_saved=True)[1].render('graph_from_dendrogram', format='jpg')
    
    #assemble graphs into a video
    #import os
    #os.system('ffmpeg -r 10 -i graph_%d.jpg -vcodec mpeg4 -y graph.mp4')
    #os.system('rm graph_*.jpg')
condmutinf(adder, (3,2))
