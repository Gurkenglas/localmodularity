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
import scipy
#torch.Tensor.repr = lambda self: self.shape.repr()
torch.Tensor.einsum = lambda self, *args, kwargs: torch.einsum(args[0], self, *args[1:], kwargs)
torch.Tensor.svdvals = lambda self: torch.linalg.svdvals(self)
torch.set_default_dtype(torch.float64)
torch.Tensor.eye_like = lambda t: torch.eye(t.shape[-1]).expand_as(t)
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
    return (covariance / np.outer(v, v)).nan_to_num(0)

def printallnondiagonalbigs(corr, threshold):
    corrbigvalues = torch.where(torch.abs(corr) > threshold, 1., 0.)
    corrbigindices = corrbigvalues.nonzero()
    for i in range(corrbigindices.__len__()):
        if corrbigindices[i,0] < corrbigindices[i,1]:
            print(corrbigindices[i]) 
            print(corr[corrbigindices[i,0], corrbigindices[i,1]])

def condmutinf(f):
    "Calculate the conditional mutual information between the two groups conditional on the input plus noise. First shape is batchsize."
    batch = f.data[0]
    resolution = 0 #setting this too low will cause the mutual information to be dominated by the noise, too high will cause the entropy to be apparently proportional to the size of the cluster
    jac = torch.autograd.functional.jacobian(lambda x: tuple(t.sum(0).reshape(-1) for t in f(x)), batch)
    colors = [plt.cm.tab10(i/len(jac)) for i,j in enumerate(jac) for _ in range(j.shape[0])]
    jac = torch.concat(jac).transpose(0,1) * 2**resolution #(1+np.log(2*np.pi))??
    jac = jac.reshape(*jac.shape[0:2], -1)
    sprint(jac.shape)
    #outputsvds = torch.linalg.svd(jac[:,-522:])
    #outputkernel = outputsvds[2][:,522:] #shape 28²-10,28².
    #identity = outputkernel @ outputkernel.transpose(1,2)
    #torch.testing.assert_close(identity, identity.eye_like())
    #projecttokernel = outputkernel.transpose(1,2) @ outputkernel
    #conditioned = jac[:,:] @ projecttokernel
    #svds = torch.linalg.svd(conditioned)
    
    #can modulesize be heterogenous?
    #ohh left singular matrix block diagonal?
    #right singular vectors represent axes, means interact badly with this
    u,d,v = torch.linalg.svd(jac[:,-10:].reshape(jac.shape[0],10,1,jac.shape[-1]), full_matrices = False) #batchsize, modulecount, modulesize=1, inputsize
    #fig, axes = plt.subplots(10,2,figsize=(20,20), sharex=True, sharey=True) # modulecount, modulesize
    #u,d,v = (x.mean(0) for x in (u,d,v)) #modulecount, modulesize, inputsize
    #for row,moduleu,moduled,modulev in tqdm(zip(axes,u,d,v)): #modulesize, inputsize
    #    for cell,leftsingularvector,singularvalue,rightsingularvector in zip(row,moduleu,moduled,modulev): #inputsize
    #        cell.imshow(rightsingularvector.reshape(28,28), cmap='plasma')
    #        cell.set_title(f'{singularvalue:.2f}')
    fig, axes = plt.subplots(20,5,figsize=(20,20), sharex=True, sharey=True) # 2*batchprefix, modulecount
    u,d,v = (torch.stack([x[:i+1].mean(0) for i in range(len(x))], 0) for x in (u,d,v)) #batchprefix, modulecount, modulesize=1, inputsize
    for row,moduleu,moduled,modulev in tqdm(zip(axes[0:10],u,d,v)): #modulecount, modulesize=1, inputsize
        for cell,leftsingularvector,singularvalue,rightsingularvector in zip(row,moduleu,moduled,modulev): #modulesize=1, inputsize
            cell.imshow(rightsingularvector[0].reshape(28,28), cmap='plasma')
            cell.set_title(f'{singularvalue[0]:.2f}')
    u,d,v = torch.linalg.svd(jac[:,-10:].reshape(jac.shape[0],10,1,jac.shape[-1]), full_matrices = False) #batchsize, modulecount, modulesize=1, inputsize
    v = v * torch.einsum("bcsi,bcsi -> bcs", v, v[:1].expand_as(v)).sign()
    u,d,v = (torch.stack([x[:i+1].mean(0) for i in range(len(x))], 0) for x in (u,d,v)) #batchprefix, modulecount, modulesize=1, inputsize
    for row,moduleu,moduled,modulev in tqdm(zip(axes[10:],u,d,v)): #modulecount, modulesize=1, inputsize
        for cell,leftsingularvector,singularvalue,rightsingularvector in zip(row,moduleu,moduled,modulev): #modulesize=1, inputsize
            cell.imshow(rightsingularvector[0].reshape(28,28), cmap='plasma')
            cell.set_title(f'{singularvalue[0]:.2f}')
    plt.show()
    u,d,v = torch.linalg.svd(jac[:,-10:].reshape(jac.shape[0],10,1,jac.shape[-1]), full_matrices = False) #batchsize, modulecount, modulesize=1, inputsize
    pi_i = np.pi*1j
    def lse(*args): #torch.Tensor.lse = lse
        t = torch.stack(args)
        tmax = max(t.real)
        return t.subtract(tmax).exp().sum().log().add(tmax)
    entr = lambda c: torch.linalg.svdvals(c.jac).add(torch.tensor(1j)).log().real.sum(1).mean().mul(2)
    #mutinf = lambda c: lse(sum(d.entr for d in c) ,c.entr+pi_i).real #-c.entr #/c.entr
    mutinf = lambda c: lse(*(d.entr for d in c) ,c.entr+pi_i).real #-c.entr #/c.entr
    bound = lambda ac,bc,c: lse(ac.entr,bc.entr,c.entr+pi_i).real # det(A+B+C) >= det(A+C) + det(B+C) - det(C)

    clusterlist = []
    count = 0   
    def iplusplus(t):
        nonlocal count
        t.index = count
        count += 1
        clusterlist.append(t)
        #t.__repr__ = lambda: f'{t.index}'
        return t
    
    class Cluster(set):
        def reify(self):
            self.jac = torch.concat([c.jac for c in self],1)
            self.indices = [i for c in self for i in c.indices]
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
        return Cluster((a,b))

    leaves = jac.unsqueeze(1).unbind(2)
    clusters = set([iplusplus(t) for t in leaves])
    for l in leaves:
        l.jac = l
        l.__hash__ = lambda: l.index
        l.entr = entr(l)
        l.indices = [l.index]
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

    plt.figure()
    batch = torch.concat(f.data,0)
    activations = torch.concat([t.reshape(t.shape[0],-1) for t in f(batch)],1) # 10000,522

    #find two images with big difference on the one child cluster and small difference on the other child cluster, and vice versa.
    def illustrate(ab, name):
        if isinstance(ab, torch.Tensor):
            acts = activations[:,ab.index]
            acts = acts.expand(acts.shape[0],acts.shape[0])
            return (acts-acts.T).pow(2)
        a,b = ab
        adists, bdists = illustrate(a, name + "a"), illustrate(b, name + "b")
        versus = adists-bdists
        min, max = torch.argmin(versus), torch.argmax(versus)
        minimage = torch.cat([batch[min//batch.shape[0]], batch[min%batch.shape[0]]],1)
        maximage = torch.cat([batch[max//batch.shape[0]], batch[max%batch.shape[0]]],1)
        illustrations = torch.cat([minimage, maximage],2)[0]
        plt.imshow(illustrations.numpy())
        plt.savefig("illustrations/" + name + ".jpg")
        return adists+bdists
    
    illustrate(ab, "_")

if False:
    adder.data = torch.rand((40,1,2,2))
    condmutinf(adder)

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
    with nested(*[hook(m) for m in model if isinstance(m, torch.nn.Linear)]):#model.modules()
        model(input)
    return tuple(all_tensors)

with torch.no_grad():
    weights = diskcache(lambda: torchexample.train_network().state_dict())()
    model = torchexample.NeuralNetwork()
    model.load_state_dict(weights)
    f = functools.partial(forward_all,model)
    f.data = list(map(lambda xy:xy[0], DataLoader(datasets.MNIST(root="data",train=False,download=True,transform=ToTensor()),batch_size=10)))
    condmutinf(f)