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

def condmutinf(f, shape):
    "Calculate the conditional mutual information between the two groups conditional on the input plus noise"

    # A differentiable function is locally linear, so we pretend that that we're analyzing a linear function. We average over multiple points.
    # How *do* we analyze a linear function? We watch how it transforms measures on the input.
    # Every space is assumed to come equipped with a reference measure. We fix it to the Lebesgue measure L for the input space and then use the pushforward measures for the other spaces.
    # On the level of data and not spaces, we also work with the distribution Omega that the training data was sampled from.
    # Pushing Omega (and L) through the linear function produces an activation distribution. Its entropy (relative to its reference measure) is the same as that of Omega, so long as the function is injective.
    # Now we can project activation space onto subsets of its dimensions, and calculate mutual information.

    jacs = []
    for i in tqdm(range(1)):
        jacs.append(torch.autograd.functional.jacobian(lambda x: torch.stack(networkgraph.make_dot(f(x), show_saved=True)[0]), torch.rand(*shape)).reshape(-1, prod(shape)))
    jac = torch.stack(jacs)
    print("jac.shape[1]: " , jac.shape[1])

    count = 0   
    def cluster_(t):
        nonlocal count
        t.index = count
        count += 1
        return t
    
    mutinfs = []
    @functools.cache
    def entr(m): 
        sings = torch.linalg.svd(jac[:,m>=1,:])[1].log()
        return (sings[:,sings[0]>-10] + (1+np.log(2*np.pi))).sum(1).mean()

    def mutinf(s):
        a,b  = sorted(list(s), key = lambda t: t.index)
        return (entr(a) + entr(b))/entr(a+b)

    leaves = torch.eye(jac.shape[1]).unbind()
    clusters = set([cluster_(t) for t in leaves])
    linkage = []        
    while len(clusters)>1:
        s = max(itertools.combinations(clusters, 2), key=mutinf)
        a,b = s
        clusters.remove(a)
        clusters.remove(b)
        c = cluster_(a+b)
        clusters.add(c)

        linkage.append([a.index, b.index, entr(c), c.count_nonzero().item()])
        mutinfs.append([((a-b).numpy()), mutinf(s).item()]) 

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
condmutinf(adder, (2,2))
