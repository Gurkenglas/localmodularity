import types

class Factory(object):
    def __init__(self, init):
        self.produce = init
    def __getattr__(self, name):
        def method(*args, **kwargs):
            produce = self.produce
            def producemore():
                s = produce()
                getattr(s,name)(*args, **{k:v() if isinstance(v, types.LambdaType) else v for k,v in kwargs.items()})
                return s
            self.produce = producemore
        return method

class Foo(object):
    def bar(self,quu):
        print(quu)

fac = Factory(Foo)
fac.bar(quu = lambda: next(ns))
fac.bar(quu = lambda: next(ns))
ns = iter([1,2,3,4])
fac.produce()
fac.produce()