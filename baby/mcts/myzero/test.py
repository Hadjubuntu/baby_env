import cloudpickle
import pickle


class Car:
    def __init__(self):
        self.t = 0


a=Car()
a.t = 3.0

a_s = cloudpickle.dumps(a)

a.t = 123.0

a_r = pickle.loads(a_s)
print(a_r.t)
