from baseline.base1.train import BaseLine1
from baseline.base2.train import BaseLine2
from baseline.base3.train import BaseLine3
from baseline.base4.train import BaseLine4
from baseline.base5.train import BaseLine5
from baseline.base6.train import BaseLine6
from baseline.base7.train import BaseLine7
from baseline.base9.train import BaseLine9

if __name__ == '__main__':
    baseline3 = BaseLine3()
    baseline3.re_run()

    baseline2 = BaseLine2()
    baseline2.re_run()

    baseline5 = BaseLine5()
    baseline5.re_run()

    baseline6 = BaseLine6()
    baseline6.re_run()

    baseline7 = BaseLine7()
    baseline7.re_run()

    baseline9 = BaseLine9()
    baseline9.re_run()


    baseline4 = BaseLine4()
    baseline4.re_run()

    baseline1 = BaseLine1()
    baseline1.re_run()


