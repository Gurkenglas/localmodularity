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

def condmutinf(f, shape):
    "Calculate the conditional mutual information between the two groups conditional on the input plus noise. First shape is batchsize."

    # A differentiable function is locally linear, so we pretend that that we're analyzing a linear function. We average over multiple points.
    # How *do* we analyze a linear function? We watch how it transforms measures on the input.
    # Every space is assumed to come equipped with a reference measure. We fix it to the Lebesgue measure L for the input space and then use the pushforward measures for the other spaces.
    # On the level of data and not spaces, we also work with the distribution Omega that the training data was sampled from.
    # Pushing Omega (and L) through the linear function produces an activation distribution. Its entropy (relative to its reference measure) is the same as that of Omega, so long as the function is injective.
    # Now we can project activation space onto subsets of its dimensions, and calculate mutual information.

    jac = diskcache(torch.autograd.functional.jacobian)(lambda x: torch.concat(tuple(map(lambda t: t.sum(0).reshape(-1), tqdm(f(x))))), torch.rand(*shape))
    jac = jac.transpose(0,1)
    sprint(jac.shape)

    count = 0   
    def cluster_(t):
        nonlocal count
        t.index = count
        count += 1
        return t
    
    mutinfs = []
    @functools.cache
    def entr(m): 
        sings = torch.linalg.svdvals(jac[:,m>=1,:]).log()
        return (sings[:,sings[0]>-10] + (1+np.log(2*np.pi))).sum(1).mean()

    # cov = jac @ jac.transpose(1,2)
    # @functools.cache
    # def entrcov(m):
    #     sparsemask = m.to_sparse()
    #     sparse2dmask = sparsemask * sparsemask.transpose(0,1)
    #     #block = cov[:,m>=1,:][:,:,m>=1]
    #     block = cov.sparse_mask(sparse2dmask)
    #     return (block+eye_like(block)*(np.e**(-20))).logdet().add(m.sum()*20).div(2).mean()

    @functools.cache
    def entr2(m):
        return torch.linalg.svdvals(jac[:,m>=1,:]).log2().add(torch.tensor(15)).maximum(torch.tensor(0)).sum(1).mean()

    def mutinf(a,b):
        ab = a+b
        #ab.entr2 = torch.linalg.svdvals(jac[:,m>=1,:]).log2().add(torch.tensor(15)).maximum(torch.tensor(0)).sum(1).mean()
        m = -(entr2(a) + entr2(b))/entr2(ab)
        m.a = a
        m.b = b
        m.ab = ab
        return m

    leaves = torch.eye(jac.shape[1]).unbind()
    #for leaf in leaves:
    clusters = set([cluster_(t) for t in leaves])
    linkage = []
    import heapq
    heap = [mutinf(a,b) for a,b in tqdm(itertools.combinations(clusters,2))]
    heapq.heapify(heap)
    
    def heapgenerator(heap):
        while heap:
            yield heapq.heappop(heap)
    
    for m in tqdm(heapgenerator(heap)):
        if hasattr(m.a, 'parent') or hasattr(m.b, 'parent'):
            continue
        clusters.remove(m.a)
        clusters.remove(m.b)
        c = cluster_(m.ab)
        m.a.parent = c
        m.b.parent = c
        for d in tqdm(clusters):
            heapq.heappush(heap, mutinf(c,d))
        clusters.add(c)
        linkage.append([m.a.index, m.b.index, entr2(m.ab),c.count_nonzero().item()])
        mutinfs.append([*((m.a-m.b).numpy()), -m.item()])

    plt.figure()    
    dn = hierarchy.dendrogram(linkage)
    leaflist = dn["leaves"]
    #Sort the leaves by their index
    for i,l in enumerate(leaflist):
        plt.plot(10*(i+0.5), entr2(leaves[l]), "o")
    plt.savefig("dendrogram.jpg")
    plt.show()

    mutinfs = [[*t[leaflist],m] for t,m in mutinfs]
    jac = jac[:,leaflist,:]
    cov = jac @ jac.transpose(1,2) #convert to csv using ([^\n\-0-9\.,]|(?<!\],)\n|(?<=\]),)
    import csv
    with open("mutinf.csv", "w", newline ='') as file:
        writer = csv.writer(file)
        writer.writerows(mutinfs)
    
    colorlist = (mpl.colors.to_hex(dn["leaves_color_list"][leaflist.index(i)]) for i in range(jac.shape[1]))
    exampleinput = torch.rand(*shape, requires_grad=True)
    networkgraph.make_dot(f(exampleinput), show_saved=True, mask = colorlist)[1].render('graph_from_dendrogram', format='jpg')
    
    #assemble graphs into a video
    #import os
    #os.system('ffmpeg -r 10 -i graph_%d.jpg -vcodec mpeg4 -y graph.mp4')
    #os.system('rm graph_*.jpg')
#condmutinf(adder, (2,2))

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

condmutinf(lambda i: forward_all(model, i), (1, 28*28))
from line_profiler import LineProfiler