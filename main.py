import functools
import itertools
from math import prod
import os
import torch
torch.set_printoptions(precision=3)
import matplotlib.pyplot as plt
import matplotlib as mpl
import torchviz
import networkgraph
import torchexample
import torchinfo
import numpy as np
from tqdm import tqdm
from scipy.cluster import hierarchy
#torch.Tensor.repr = lambda self: self.shape.repr()
torch.Tensor.einsum = lambda self, *args, kwargs: torch.einsum(args[0], self, *args[1:], kwargs)
torch.set_default_dtype(torch.float64)
import jax

def eye_like(tensor):
    return torch.eye(tensor.shape[-1])

import varname
def sprint(x):
    print(varname.nameof(x,frame=2,vars_only=False), x)

def diskcache(f):
    "Doesn't check args, only caches tensors."
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        name = varname.varname()
        filepath = name+'.pt'
        sprint(filepath)
        if os.path.exists(filepath):
            return torch.load(filepath)
        else:
            result = f(*args, **kwargs)
            torch.save(result, filepath)
            return result
    return wrapper

def halfadder(a,b):
    return a*b, (1-a)*b+(1-b)*a

def adder(ins):
    ins = ins.permute(1,2,0)
    sum = []
    carry = 0
    activations = []
    for a,b in ins:
        u,v = halfadder(a,b)
        w,x = halfadder(v,carry)
        carry= u+w
        sum.append(x)
        activations += [a,b,u,v,w,x, carry]
    sum.append(carry)
    return torch.stack(activations)

def condmutinf(f, shape):
    "Calculate the conditional mutual information between the two groups conditional on the input plus noise. First shape is batchsize."

    # A differentiable function is locally linear, so we pretend that that we're analyzing a linear function. We average over multiple points.
    # How *do* we analyze a linear function? We watch how it transforms measures on the input.
    # Every space is assumed to come equipped with a reference measure. We fix it to the Lebesgue measure L for the input space and then use the pushforward measures for the other spaces.
    # On the level of data and not spaces, we also work with the distribution Omega that the training data was sampled from.
    # Pushing Omega (and L) through the linear function produces an activation distribution. Its entropy (relative to its reference measure) is the same as that of Omega, so long as the function is injective.
    # Now we can project activation space onto subsets of its dimensions, and calculate mutual information.

    jac = torch.autograd.functional.jacobian(lambda x: torch.concat(tuple(map(lambda t: t.sum(0).reshape(-1), tqdm(f(x))))), torch.rand(*shape))
    jac = jac.transpose(0,1) * 2**20 #(1+np.log(2*np.pi))??
    jac = jac.reshape(*jac.shape[0:2], -1)
    sprint(jac.shape)

    count = 0   
    def cluster_(t):
        nonlocal count
        t.index = count
        count += 1
        return t

    class Pair:
        @functools.cache
        def __init__(self,ab,c):
            if hasattr(ab,'a'):
                ac,bc = Pair(ab.a,c), Pair(ab.b,c)
                # det(A+B+C) >= det(A+C) + det(B+C) - det(C)
                assert ab.a == ac.a and ac.b == bc.b and ab.b == bc.a
                a,b,c=ac.a,bc.a,ac.b
                self.a,self.b=ab,c
                foo = lambda ac,bc,c: torch.logsumexp(torch.stack([ac()+c(),bc()+c(),0]), dim=0)-c()
                self.__call__ = lambda self: max([foo(ac,bc,c),foo(ab,bc,b),foo(ac,ab,a)])
            else:
                self.a,self.b=ab,c
                self.__call__ = lambda self: torch.logsumexp(torch.stack([ab()+c(),0]), dim=0)-c()
        def reify(self):
            # this is logdet(I + J@J^T) = logdet(I + J^T@J)
            entr = torch.linalg.svdvals(jac[:,m>=1,:]).add(torch.tensor(1)).log().sum(1).mean()
            self.__call__ = lambda self: entr
            return entr

    def mutinf(a,b): # a was just assembled from two clusters
        ab = Pair(a,b) #FIXME
        m = ab()-a()-b()
        m.a = a
        m.b = b
        m.ab = ab
        m.actualmutinf = lambda: entr(ab)-entr(a)-entr(b)
        #ab.entr = torch.linalg.svdvals(jac[:,m>=1,:]).log2().add(torch.tensor(15)).maximum(torch.tensor(0)).sum(1).mean()
        #m = -(entr(a) + entr(b))/entr(ab)
        return m

    leaves = torch.eye(jac.shape[1]).unbind()
    clusters = set([cluster_(t) for t in leaves])
    linkage = []
    mutinfs = []
    import heapq
    heap = [mutinf(a,b) for a,b in tqdm(itertools.combinations(clusters,2))]
    heapq.heapify(heap)
    
    def heapgenerator(heap):
        while heap:
            m = heapq.heappop(heap)
            if hasattr(m.a, 'parent') or hasattr(m.b, 'parent'):
                continue
            yield m
    
    for m in tqdm(heapgenerator(heap)):
        clusters.remove(m.a)
        clusters.remove(m.b)
        ab = cluster_(m.ab)
        m.a.parent = ab
        m.b.parent = ab
        linkage.append([m.a.index, m.b.index, ab().reify(),ab.count_nonzero().item()])
        mutinfs.append([((m.a-m.b).numpy()), -m.item()])
        for c in clusters:
            newpairs = Pair(ab.a,c)
            heapq.heappush(heap, mutinf(ab,c))
        clusters.add(ab)

    plt.figure()    
    dn = hierarchy.dendrogram(linkage)
    leaflist = dn["leaves"]
    #Sort the leaves by their index
    for i,l in enumerate(leaflist):
        plt.plot(10*(i+0.5), entr(leaves[l]), "o")
    plt.savefig("dendrogram.jpg")
    plt.show()

    mutinfs = [[*t[leaflist],m] for t,m in mutinfs]
    jac = jac[:,leaflist,:]
    cov = jac @ jac.transpose(1,2)
    import csv
    with open("covmutinf.csv", "w", newline ='') as file:
        writer = csv.writer(file)
        writer.writerows(cov)
        writer.writerows(mutinfs)
    
    #colorlist = [mpl.colors.to_hex(dn["leaves_color_list"][leaflist.index(i)]) for i in range(jac.shape[1])]
    exampleinput = torch.rand(*shape, requires_grad=True)
    torchviz.make_dot(f(exampleinput)).render('graph_from_dendrogram', format='jpg')
    
    #assemble graphs into a video
    #import os
    #os.system('ffmpeg -r 10 -i graph_%d.jpg -vcodec mpeg4 -y graph.mp4')
    #os.system('rm graph_*.jpg')
condmutinf(adder, (10,30,2))

weights = diskcache(lambda: torchexample.train_network().state_dict())()
model = torchexample.NeuralNetwork()
model.load_state_dict(weights)

#define a function that wraps a model and given an input returns not the output but a tensor of all numbers computed within the model
def forward_all(model, input):
    all_tensors = (input.reshape(-1),)
    def hook(model, input, output):
        nonlocal all_tensors
        all_tensors += (output,)
    for layer in model.children():
        layer.register_forward_hook(hook)
    model(input)
    for layer in model.children():
        layer.register_forward_hook(None)
    return all_tensors

#condmutinf(lambda i: forward_all(model, i), (1, 28*28))
#from line_profiler import LineProfiler