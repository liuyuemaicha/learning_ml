# coding:utf8
import numpy as np
import random


def gen_line_data(sample_num=100):
    """
    y = 3*x1 + 4*x2
    :return:
    """
    x1 = np.linspace(0, 9, sample_num)
    x2 = np.linspace(4, 13, sample_num)
    x = np.concatenate(([x1], [x2]), axis=0).T
    y = np.dot(x, np.array([3, 4]).T)  # y 列向量
    return x, y


def bgd(samples, y, step_size=0.01, max_iter_count=10000):
    """
    批量梯度下降
    :param samples:
    :param y:
    :param step_size:
    :param max_iter_count:
    :return:
    """
    sample_num, dim = samples.shape
    y = y.flatten()
    loss = 10
    iter_count = 0
    # w = np.ones((dim,), dtype=np.float32)
    w = np.ones((1, dim), dtype=np.float32)
    while loss > 0.001 and iter_count < max_iter_count:
        # loss = 0
        # error = np.zeros((sample_num, dim), dtype=np.float32)
        predict_y = np.multiply(w, samples)
        predict_y = np.sum(predict_y, axis=1)
        y = np.expand_dims(y, axis=1)
        predict_y = np.expand_dims(predict_y, axis=1)
        error = (predict_y - y) * samples
        error = np.average(error, axis=0)
        w -= error*step_size
        predict_y = np.multiply(w, samples)
        predict_y = np.sum(predict_y, axis=1)
        y = y.flatten()
        loss = np.power((predict_y - y), 2)
        loss = np.sum(loss)
        loss = loss / (sample_num * dim)
        print("iter_count: ", iter_count, "the loss:", loss)
        iter_count += 1
    return w


def sgd(samples, y, step_size=0.01, max_iter_count=10000):
    """
    随机梯度下降法
    :param samples: 样本
    :param y: 结果value
    :param step_size: 每一接迭代的步长
    :param max_iter_count: 最大的迭代次数
    :param batch_size: 随机选取的相对于总样本的大小
    :return:
    """
    sample_num, dim = samples.shape
    y = y.flatten()
    w = np.ones(dim, dtype=np.float32)

    loss = 10
    iter_count = 0
    while loss > 0.001 and iter_count < max_iter_count:
        random_index = np.random.randint(1, sample_num, 1)[0]

        predict_y = np.dot(w, samples[random_index])
        error = (predict_y - y[random_index]) * samples[random_index]
        w -= step_size * error

        predict_y = np.multiply(np.expand_dims(w, 0), samples)
        predict_y = np.sum(predict_y, axis=1)
        loss = np.power((predict_y - y), 2)
        loss = np.sum(loss) / (sample_num * dim)
        print("iter_count: ", iter_count, "the loss:", loss)

        iter_count += 1
    return w


def mbgd(samples, y, batch_size=10, step_size=0.01, max_iter_count=10000):
    """
    MBGD（Mini-batch gradient descent）小批量梯度下降：每次迭代使用b组样本
    :param samples:
    :param y:
    :param step_size:
    :param max_iter_count:
    :param batch_size:
    :return:
    """
    sample_num, dim = samples.shape
    y = y.flatten()
    w = np.ones((1, dim), dtype=np.float32)

    loss = 10
    iter_count = 0
    while loss > 0.001 and iter_count < max_iter_count:
        random_index = np.random.randint(1, sample_num, batch_size)
        batch_samples = samples[random_index]
        batch_y = y[random_index]

        predict_y = np.multiply(w, batch_samples)
        predict_y = np.sum(predict_y, axis=1)
        error = np.expand_dims((predict_y - batch_y), 1) * batch_samples
        error = np.sum(error, axis=0)
        w -= error / float(batch_size) * step_size

        predict_y = np.multiply(w, samples)
        predict_y = np.sum(predict_y, axis=1)
        loss = np.power((predict_y - y), 2)
        loss = np.sum(loss) / (sample_num * dim)
        print("iter_count: ", iter_count, "the loss:", loss)

        iter_count += 1
    return w


if __name__ == '__main__':
    samples, y = gen_line_data(100)
    # w = bgd(samples, y)
    # print(w)  # 会很接近[3, 4]

    # w = sgd(samples, y)
    # print(w)  # 会很接近[3, 4]

    w= mbgd(samples, y)
    print (w)