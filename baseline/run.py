from baseline.base1.train import BaseLine1
from baseline.base2.train import BaseLine2
from baseline.base3.train import BaseLine3
from baseline.base4.train import BaseLine4

if __name__ == '__main__':
    baseline3 = BaseLine3()
    baseline3.run()

    baseline2 = BaseLine2()
    baseline2.run()

    baseline1 = BaseLine1()
    baseline1.run()

    baseline4 = BaseLine4()
    baseline4.run()

