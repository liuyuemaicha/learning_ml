# coding:utf8
import numpy as np

# f(x, y) = (1-x)**2 + 100*(y-x**2)**2   search x, y for f(x,y) = 0
def cal_rosenbrock(x1, x2):
    """
    计算rosenbrock函数的值
    :param x1:
    :param x2:
    :return:
    """
    return (1 - x1) ** 2 + 100 * (x2 - x1 ** 2) ** 2


def cal_rosenbrock_prax(x1, x2):
    """
    对x1求偏导
    """
    return -2 + 2 * x1 - 400 * (x2 - x1 ** 2) * x1


def cal_rosenbrock_pray(x1, x2):
    """
    对x2求偏导
    """
    return 200 * (x2 - x1 ** 2)


def for_rosenbrock_func(max_iter_count=100000, step_size=0.001):
    pre_x = np.zeros((2,), dtype=np.float32)
    # pre_x = np.random.randn(2)
    print pre_x
    loss = 1
    iter_count = 0
    while loss > 0.001 and iter_count < max_iter_count:
        error = np.zeros((2,), dtype=np.float32)
        error[0] = cal_rosenbrock_prax(pre_x[0], pre_x[1])
        error[1] = cal_rosenbrock_pray(pre_x[0], pre_x[1])
        print error
        pre_x = pre_x - error * step_size
        print pre_x
        loss = cal_rosenbrock(pre_x[0], pre_x[1])  # 最小值为0  loss = loss - 0
        print("iter_count: ", iter_count, "the loss:", loss)
        iter_count += 1
    return pre_x


if __name__ == '__main__':
    w = for_rosenbrock_func()
    print(w)
