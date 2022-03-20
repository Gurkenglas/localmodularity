import functools
import itertools
import torch
torch.set_printoptions(precision=3)
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.cluster import hierarchy
import heapq
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
torch.Tensor.svd = lambda self, *args, **kwargs: torch.linalg.svd(self, *args, **kwargs)
#torch.Tensor.repr = lambda self: self.shape.repr()
torch.Tensor.einsum = lambda self, *args, **kwargs: torch.einsum(args[0], self, *args[1:], **kwargs)
torch.Tensor.svdvals = lambda self: torch.linalg.svdvals(self)
torch.set_default_dtype(torch.float64)
torch.Tensor.eye_like = lambda t: torch.eye(t.shape[-1]).expand_as(t)
import varname
sprint = lambda *args: [print(varname.nameof(a,frame=3,vars_only=False), a) for a in args]

torch.set_grad_enabled(False)
edgedetector = torch.nn.Conv2d(1,1,2,padding=1)
edgedetector.weight.data = torch.tensor([[[[1.,-1.],[1.,-1.]]]])
edgedetector.bias.data = torch.tensor([0.])
f = edgedetector
f.data = list(map(lambda xy:xy[0], DataLoader(datasets.MNIST(root="data",train=False,download=True,transform=ToTensor()),batch_size=1)))
batch = f.data[0]
resolution = 0 #setting this too low will cause the mutual information to be dominated by the noise, too high will cause the entropy to be apparently proportional to the size of the cluster
jac = torch.autograd.functional.jacobian(lambda x: tuple(t.sum(0).reshape(-1) for t in f(x)), batch)
colors = [plt.cm.tab10(i/len(jac)) for i,j in enumerate(jac) for _ in range(j.shape[0])]
jac = torch.concat(jac).transpose(0,1) * 2**resolution #(1+np.log(2*np.pi))??
jac = jac.reshape(*jac.shape[0:2], -1)
sprint(jac.shape)
entr = lambda jac: torch.linalg.svdvals(jac).add(torch.tensor(1j)).log().real.sum(1).mean().mul(2)
mutinf = lambda c: sum(d.entr for d in c)-2*c.entr

clusterlist = []
count = 0
def iplusplus(t):
    global count
    t.index = count
    count += 1
    clusterlist.append(t)
    #t.__repr__ = lambda: f'{t.index}'
    return t
class Cluster(set):
    __lt__ = lambda self,other: self.mutinf > other.mutinf #maxheap
    __repr__ = lambda self: f'({self[0].index},{self[1].index})'
    __hash__ = lambda self: self.index # invariant: not called before index exists

def cachedpair(a,b):
    if a.index > b.index: a,b = b,a
    return cachedset(a,b)

@functools.cache
def cachedset(a,b):
    return Cluster((a,b))

leaves = jac[:,:29*8].unsqueeze(1).unbind(2)
clusters = set([iplusplus(t) for t in leaves])
for l in leaves:
    l.jac = l
    l.__hash__ = lambda: l.index
    l.entr = l.exactentr = entr(l)
    l.indices = [l.index]
heap = [cachedpair(a,b) for a,b in tqdm(itertools.combinations(clusters,2))]
for ab in heap:
    ab.entr = 0 # max(l.entr for l in ab)
    ab.mutinf = torch.inf # sum(l.entr for l in ab)
heapq.heapify(heap)

def heapgenerator(heap):
    while heap:
        ab = heapq.heappop(heap)
        if any(c.jac is None for c in ab): continue
        if hasattr(ab,'exactentr'):
            yield ab
        else:
            ab.jac = torch.concat([c.jac for c in ab],1)
            ab.indices = [i for c in ab for i in c.indices]
            ab.exactentr = entr(ab.jac)
            assert ab.entr <= ab.exactentr
            ab.entr = ab.exactentr
            ab.mutinf = mutinf(ab)
            heapq.heappush(heap, ab)

linkage = []
for ab in tqdm(heapgenerator(heap)):
    a,b = ab # a and b are fungible
    clusters.remove(a), clusters.remove(b)
    iplusplus(ab)
    ab.jac = torch.concat([a.jac,b.jac],1)
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
image = torch.zeros(29*8)
def color(ab,strength):
    if isinstance(ab, torch.Tensor): return
    a,b = ab
    for i in b.indices:
        image[i] += strength
    color(a,strength/2)
    color(b,strength/2)
color(ab,1.)
plt.imshow(image.reshape(-1,29), cmap='gray')
plt.show()

plt.figure()
dn = hierarchy.dendrogram(linkage)
leaflist = dn["leaves"]
for i,l in enumerate(leaflist):
    plt.plot(10*(i+0.5), leaves[l].entr, "o", color=colors[leaves[l].index])
plt.savefig("dendrogram.jpg")
plt.show()