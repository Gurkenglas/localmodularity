import contextlib
import functools
import itertools
from math import prod
import os
from typing import Callable
import torch
torch.set_printoptions(precision=3)
import matplotlib.pyplot as plt
import torchexample
import numpy as np
from tqdm import tqdm
from scipy.cluster import hierarchy
import heapq
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
#torch.Tensor.repr = lambda self: self.shape.repr()
torch.Tensor.einsum = lambda self, *args, kwargs: torch.einsum(args[0], self, *args[1:], kwargs)
torch.Tensor.svdvals = lambda self: torch.linalg.svdvals(self)
torch.set_default_dtype(torch.float64)
eye_like = lambda t: torch.eye(t.shape[-1])
Object = lambda **kwargs: type("Object", (), kwargs) # Anonymous objects! :)
import varname
sprint = lambda *args: [print(varname.nameof(a,frame=3,vars_only=False), a) for a in args]

def diskcache(f):
    "Doesn't check args, only caches tensors."
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        name = varname.varname()
        filepath = name+'.pt'
        if os.path.exists(filepath): return torch.load(filepath)
        result = f(*args, **kwargs)
        torch.save(result, filepath)
        return result
    return wrapper

halfadder = lambda a,b: (a*b, (1-a)*b+(1-b)*a)
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

def correlation_from_covariance(covariance): #cov = jac @ jac.transpose(1,2)
    covariance = covariance[0] #throw away batches, this is only for debugging
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation

def printallnondiagonalbigs(corr, threshold):
    corrbigvalues = torch.where(torch.abs(corr) > threshold, 1., 0.)
    corrbigindices = corrbigvalues.nonzero()
    for i in range(corrbigindices.__len__()):
        if corrbigindices[i,0] < corrbigindices[i,1]:
            print(corrbigindices[i]) 
            print(corr[corrbigindices[i,0], corrbigindices[i,1]])

def condmutinf(f, shape):
    "Calculate the conditional mutual information between the two groups conditional on the input plus noise. First shape is batchsize."

    # A differentiable function is locally linear, so we pretend that that we're analyzing a linear function. We average over multiple points.
    # How *do* we analyze a linear function? We watch how it transforms measures on the input.
    # Every space is assumed to come equipped with a reference measure. We fix it to the Lebesgue measure L for the input space and then use the pushforward measures for the other spaces.
    # On the level of data and not spaces, we also work with the distribution Omega that the training data was sampled from.
    # Pushing Omega (and L) through the linear function produces an activation distribution. Its entropy (relative to its reference measure) is the same as that of Omega, so long as the function is injective.
    # Now we can project activation space onto subsets of its dimensions, and calculate mutual information.
    testinput = torch.rand(*shape)
    testoutput = [t.sum(0).reshape(-1) for t in f(testinput)]
    #get an exampleinput from the dataset of MNIST
    train_dataloader,test_dataloader = (
        DataLoader(datasets.MNIST(root="data",train=b,download=True,transform=ToTensor()),batch_size=shape[0]) for b in (True, False)
        )
    testinput = next(iter(test_dataloader))[0]
    resolution = 0 #setting this too low will cause the mutual information to be dominated by the noise, too high will cause the entropy to be apparently proportional to the size of the cluster
    jac = torch.autograd.functional.jacobian(lambda x: tuple(t.sum(0).reshape(-1) for t in f(x)), testinput)
    colors = [plt.cm.tab10(i/len(jac)) for i,j in enumerate(jac) for _ in range(j.shape[0])]
    jac = torch.concat(jac).transpose(0,1) * 2**(resolution) #(1+np.log(2*np.pi))??
    jac = jac.reshape(*jac.shape[0:2], -1)
    sprint(jac.shape)
    
    def lse(*args): #torch.Tensor.lse = lse
        t = torch.stack(args)
        tmax = max(t.real)
        return t.subtract(tmax).exp().sum().log().add(tmax)
    entr = lambda c: torch.linalg.svdvals(c.jac).add(torch.tensor(1j)).log().real.sum(1).mean().mul(2)
    mutinf = lambda c: sum(d.entr for d in c) -2*c.entr #-c.entr #/c.entr
    bound = lambda ac,bc,c: lse(ac.entr,bc.entr,c.entr+np.pi*1j).real # det(A+B+C) >= det(A+C) + det(B+C) - det(C)

    count = 0   
    def iplusplus(t):
        nonlocal count
        t.index = count
        count += 1
        #t.__repr__ = lambda: f'{t.index}'
        return t
    
    class Pair(set):
        def reify(self):
            self.jac = torch.concat([c.jac for c in self],1)
            self.entr = entr(self)
            self.mutinf = mutinf(self)
            self.reify = lambda: True
            return False
        __lt__ = lambda self,other: self.mutinf > other.mutinf #maxheap
        __repr__ = lambda self: f'({self[0].index},{self[1].index})'
        __hash__ = lambda self: self.index # invariant: not called before index exists
    
    def cachedpair(a,b):
        if a.index > b.index: a,b = b,a
        return cachedset(a,b)
    
    @functools.cache
    def cachedset(a,b):
        return Pair((a,b))

    leaves = jac.unsqueeze(1).unbind(2)
    clusters = set([iplusplus(t) for t in leaves])
    for l in leaves:
        l.jac = l
        l.__hash__ = lambda: l.index
        l.entr = entr(l)
    linkage = []
    heap = [cachedpair(a,b) for a,b in tqdm(itertools.combinations(clusters,2))]
    [pair.reify() for pair in tqdm(heap)]
    heapq.heapify(heap)
    
    def heapgenerator(heap):
        while heap:
            p = heapq.heappop(heap)
            if any(c.jac is None for c in p): continue
            currententr = p.entr
            if p.reify(): yield p
            else:
                assert currententr <= p.entr
                heapq.heappush(heap, p)
    
    for ab in tqdm(heapgenerator(heap)):
        a,b = ab # a and b are fungible
        clusters.remove(a), clusters.remove(b)
        iplusplus(ab)
        a.jac = b.jac = None
        linkage.append([a.index, b.index, ab.entr, ab.jac.shape[1]])
        for c in clusters:
            abc,ac,bc = cachedpair(ab,c),cachedpair(a,c),cachedpair(b,c)
            assert isinstance(c,torch.Tensor) or c.reify()
            abc.entr = max([bound(ac,bc,c),bound(ab,bc,b),bound(ac,ab,a)])
            abc.mutinf = mutinf(abc)
            heapq.heappush(heap, abc)
        clusters.add(ab)

    plt.figure()    
    dn = hierarchy.dendrogram(linkage)
    leaflist = dn["leaves"]
    for i,l in enumerate(leaflist):
        plt.plot(10*(i+0.5), leaves[l].entr, "o", color=colors[leaves[l].index])
    plt.savefig("dendrogram.jpg")
    plt.show()
#condmutinf(adder, (1,2,2))

weights = diskcache(lambda: torchexample.train_network().state_dict())()
model = torchexample.NeuralNetwork()
model.load_state_dict(weights)

def nested(*args): #why was nested deprecated??
    class Nested:
        def __enter__(self): [a.__enter__() for a in args]
        def __exit__(self,*exc): [a.__exit__() for a in args]
    return Nested()

# wrap a model to return all intermediate tensors
def forward_all(model, input):
    all_tensors = []#input]
    def hook(layer):
        h = layer.register_forward_hook(lambda module, input, output: all_tensors.append(output))
        return contextlib.closing(Object(close = lambda: h.remove()))
    with nested(*[hook(m) for m in model[2:] if isinstance(m, torch.nn.Linear)]):
        model(input)
    return tuple(all_tensors)

with torch.no_grad():
    condmutinf(lambda i: forward_all(model, i), (1, 28*28))
#from line_profiler import LineProfiler

#make dendrogram work
#optimize heap thing further
#write actual agenda code