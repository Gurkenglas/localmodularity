import contextlib
import functools
import itertools
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
torch.Tensor.svd = lambda self, *args, **kwargs: torch.linalg.svd(self, *args, **kwargs)
#torch.Tensor.repr = lambda self: self.shape.repr()
torch.Tensor.einsum = lambda self, *args, **kwargs: torch.einsum(args[0], self, *args[1:], **kwargs)
torch.Tensor.svdvals = lambda self: torch.linalg.svdvals(self)
torch.set_default_dtype(torch.float64)
torch.Tensor.eye_like = lambda t: torch.eye(t.shape[-1]).expand_as(t)
Object = lambda **kwargs: type("Object", (), kwargs) # Anonymous objects! :)
import varname
sprint = lambda *args: [print(varname.nameof(a,frame=3,vars_only=False), a) for a in args]

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
    def lse(*args): #torch.Tensor.lse = lse
        t = torch.stack(args)
        tmax = t.real.max(0)
        return t.subtract(tmax).exp().sum().log().add(tmax)
    entr = lambda c: torch.linalg.svdvals(c.jac).add(torch.tensor(1j)).log().real.sum(1).mean().mul(2)
    mutinf = lambda c: sum(d.entr for d in c)-2*c.entr

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
            self.entr = self.exactentr = entr(self)
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
        l.entr = l.exactentr = entr(l)
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

        # det(A+B+C) >= det(A+C) + det(B+C) - det(C)
        bound = torch.tensor([[a.entr, b.entr, c.exactentr, ab.entr, cachedpair(a,c).entr, cachedpair(b,c).entr] for c in clusters])
        if bound.shape == (0,): bound = torch.empty(0,f.data[0].shape[0],6)
        bound = torch.index_select(bound, dim=1, index=torch.tensor([4,5,2,3,5,1,3,4,0])).reshape(-1,3,3)
        bmax = bound.max(2,keepdim=True)[0]
        bound = bound.subtract(bmax).exp()
        bound[:,:,2] *= -1
        bound = bound.sum(2).log().add(bmax[:,:,0]).max(1)[0]

        for c,onebound in zip(clusters,bound):
            abc = cachedpair(ab,c)
            abc.entr = onebound
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

#f = functools.partial(forward_all,torchexample.get_network())
#with torch.no_grad():
    #f.data = list(map(lambda xy:xy[0], DataLoader(datasets.MNIST(root="data",train=False,download=True,transform=ToTensor()),batch_size=1)))
    #condmutinf(f)

# we are getting garbage out of our network analysis. maybe try with a pretrained generative network?
# no, let's do some simple convolutions for edge detection and treat that as the network to analyze.

edgedetector = torch.nn.Conv2d(1,1,2,padding=1)
edgedetector.weight.data = torch.tensor([[[[1.,-1.],[1.,-1.]]]])
sprint(edgedetector.bias.data)
with torch.no_grad():
    f = edgedetector
    f.data = list(map(lambda xy:xy[0], DataLoader(datasets.MNIST(root="data",train=False,download=True,transform=ToTensor()),batch_size=1)))
    condmutinf(f)