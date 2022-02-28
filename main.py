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

def condmutinf(f, shape):
    "Calculate the conditional mutual information between the two groups conditional on the input plus noise. First shape is batchsize."

    # A differentiable function is locally linear, so we pretend that that we're analyzing a linear function. We average over multiple points.
    # How *do* we analyze a linear function? We watch how it transforms measures on the input.
    # Every space is assumed to come equipped with a reference measure. We fix it to the Lebesgue measure L for the input space and then use the pushforward measures for the other spaces.
    # On the level of data and not spaces, we also work with the distribution Omega that the training data was sampled from.
    # Pushing Omega (and L) through the linear function produces an activation distribution. Its entropy (relative to its reference measure) is the same as that of Omega, so long as the function is injective.
    # Now we can project activation space onto subsets of its dimensions, and calculate mutual information.
    testinput = torch.rand(*shape)
    testoutput = tuple(map(lambda t: t.sum(0).reshape(-1),f(testinput)))

    jac = torch.autograd.functional.jacobian(lambda x: tuple(map(lambda t: t.sum(0).reshape(-1), tqdm(f(x)))), torch.rand(*shape))
    colors = [plt.cm.tab10(i/len(jac)) for i,j in enumerate(jac) for _ in range(j.shape[0])]
    jac = torch.concat(jac).transpose(0,1) * 2**20 #(1+np.log(2*np.pi))??
    jac = jac.reshape(*jac.shape[0:2], -1)
    sprint(jac.shape)
    
    def lse(*args): #torch.Tensor.lse = lse
        t = torch.stack(args)
        tmax = max(t.real)
        return t.subtract(tmax).exp().sum().log().add(tmax)
    entr = lambda c: torch.linalg.svdvals(jac[:,c.mask>=1,:]).add(torch.tensor(1j)).log().real.sum(1).mean().mul(2)
    mutinf = lambda c: (c.a.entr + c.b.entr) -2*c.entr #-c.entr #/c.entr
    bound = lambda ac,bc,c: lse(ac.entr,bc.entr,c.entr+np.pi*1j).real # det(A+B+C) >= det(A+C) + det(B+C) - det(C)

    count = 0   
    def iplusplus(t):
        nonlocal count
        t.index = count
        count += 1
        return t
    
    class Pair():
        def __init__(self,ab,c):
            self.a,self.b=ab,c
            if hasattr(ab,'a'):
                ac,bc=cachedpair(ab.a,c),cachedpair(ab.b,c)
                assert id(ab.a) == id(ac.a) and id(ac.b) == id(bc.b) and id(ab.b) == id(bc.a)
                a,b,c=ac.a,bc.a,ac.b
                self.ab,self.ac,self.bc = ab,ac,bc
                self._a,self._b,self._c = a,b,c
                assert isinstance(c,torch.Tensor) or c.reify()
                self.entr = max([bound(ac,bc,c),bound(ab,bc,b),bound(ac,ab,a)])
                self.mutinf = mutinf(self)
            else:
                self.reify()
        def reify(self):
            self.mask = self.a.mask + self.b.mask
            exact = entr(self)
            try:
                assert getattr(self,'entr',-torch.inf) <= exact
            except:
                a,b,c,ab,ac,bc = self._a,self._b,self._c,self.ab,self.ac,self.bc
                COV=lambda m:(lambda c: c+torch.eye(c.shape[-1]))(jac[0,m.mask>=1,:].T@jac[0,m.mask>=1,:]).logdet()/2
                cov=lambda m:(lambda c: c+torch.eye(c.shape[-1]))(jac[0,m.mask>=1,:]@jac[0,m.mask>=1,:].T).logdet()/2
                gross=lambda m:(lambda c: c+torch.eye(c.shape[-1]))(jac[0,m.mask>=1,:].T@jac[0,m.mask>=1,:])
                seven=lambda foo:{x_:foo(x) for x_,x in zip(["a","b","c","ab","ac","bc","abc"],[a,b,c,ab,ac,bc,self])}
                sprint([f"{x_}+{y_}={z_}+{w_}" for (x_,x),(y_,y),(z_,z),(w_,w) in itertools.combinations(seven(cov).items(),4) if torch.allclose(x+y,z+w,rtol=0.01,atol=0)])
                sprint([f"{x_}+{y_}={z_}+{w_}" for (x_,x),(y_,y),(z_,z),(w_,w) in itertools.combinations(seven(cov).items(),4) if torch.allclose(x+y,z+w,rtol=0.01,atol=0)])
                sprint(seven(gross)['abc'].det()+seven(gross)['c'].det()-seven(gross)['ac'].det()-seven(gross)['bc'].det())
                print("done") #torch.stack(list(seven(gross).values()))
            self.entr = exact
            self.mutinf = mutinf(self)
            self.reify = lambda: True
            return False
        def __lt__(self,other):
            return self.mutinf > other.mutinf #maxheap
        def __hash__(self):
            return self.index
    cachedpair = functools.cache(Pair)

    leaves = torch.eye(jac.shape[1]).unbind()
    clusters = set([iplusplus(t) for t in leaves])
    for l in leaves:
        l.mask = l
        l.__hash__ = lambda: l.index
        l.entr = entr(l)
    linkage = []
    mutinfs = []
    import heapq 
    heap = [cachedpair(a,b) for a,b in tqdm(itertools.combinations(clusters,2))]
    heapq.heapify(heap)
    
    def heapgenerator(heap):
        while heap:
            p = heapq.heappop(heap)
            if hasattr(p.a, 'parent') or hasattr(p.b, 'parent'): continue
            if p.reify(): yield p
            else: heapq.heappush(heap, p)
    
    for p in tqdm(heapgenerator(heap)):
        clusters.remove(p.a)
        clusters.remove(p.b)
        iplusplus(p)
        p.a.parent = p
        p.b.parent = p
        linkage.append([p.a.index, p.b.index, p.entr, p.mask.count_nonzero().item()])
        mutinfs.append([(p.a.mask-p.b.mask).numpy(), p.mutinf.numpy()])
        for c in clusters:
            heapq.heappush(heap, cachedpair(p,c))
        clusters.add(p)

    plt.figure()    
    dn = hierarchy.dendrogram(linkage)
    leaflist = dn["leaves"]
    for i,l in enumerate(leaflist):
        plt.plot(10*(i+0.5), leaves[l].entr, "o", color=colors[leaves[l].index])
    plt.savefig("dendrogram.jpg")
    plt.show()

    mutinfs = [[*t[leaflist],m] for t,m in mutinfs]
    jac = jac[:,leaflist,:]
    cov = jac @ jac.transpose(1,2)
    import csv
    with open("covmutinf.csv", "w", newline ='') as file:
        writer = csv.writer(file)
        writer.writerows(cov.numpy())
        writer.writerows(mutinfs)
    
    #colorlist = [mpl.colors.to_hex(dn["leaves_color_list"][leaflist.index(i)]) for i in range(jac.shape[1])]
    exampleinput = torch.rand(*shape, requires_grad=True)
    #torchviz.make_dot(f(exampleinput)).render('graph_from_dendrogram', format='jpg')
    
    #assemble graphs into a video
    #import os
    #os.system('ffmpeg -r 10 -i graph_%d.jpg -vcodec mpeg4 -y graph.mp4')
    #os.system('rm graph_*.jpg')
condmutinf(adder, (1,2,2))

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
    with nested(*[hook(m) for m in model.modules() if isinstance(m, torch.nn.Linear)]):
        model(input)
    return tuple(all_tensors)

with torch.no_grad():
    condmutinf(lambda i: forward_all(model, i), (1, 28*28))
#from line_profiler import LineProfiler

#make dendrogram work
#optimize heap thing further
#write actual agenda code