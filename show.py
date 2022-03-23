import matplotlib.pyplot as plt
import numpy as np
import pickle
import os


def show(data, max_length=10000, stride=1):
    value = data['process']
    title = data['title']
    value = np.array(value)
    x_data = value[:max_length:stride, 0]
    y_data = value[:max_length:stride, 1]
    last_values = 0
    for i in range(len(y_data)):
        y_data[i] = np.log10(y_data[i]) if y_data[i] != 0 else last_values
        last_values = y_data[i]

    plt.plot(x_data, y_data, linewidth=1, linestyle=':')
    plt.title(title)
    plt.ylabel('Benchmark function value(10^x)')
    plt.xlabel('Number of iterations')
    with open('img/%s_%s_%s.png' % (title, stride, max_length), 'wb') as f:
        plt.savefig(f)
    plt.show()


def show_position(data, max_length=1000):
    value = data['positions'][:max_length:max_length//9][:9]
    value = np.array(value)
    plt.figure(figsize=[15, 15], dpi=100)
    for i in range(len(value)):
        plt.subplot(3, 3, i+1)
        plt.title('iteration='+str(value[i][0]))
        plt.xlim(4.5, 5.5)
        plt.ylim(4.5, 5.5)
        v = value[i][1]
        x_data = v[:, 0]
        y_data = v[:, 1]
        plt.scatter(x_data, y_data, s=20, c="#ff1212", marker='o')
    with open('img_scatter/%s.png' % max_length, 'wb') as f:
        plt.savefig(f)
    plt.show()


if __name__ == '__main__':
    for root, dirs, files in os.walk('data'):
       for f in files:
           s = pickle.load(open('data_position/sphere_function.txt', 'rb'))
           show_position(s, max_length=100)
           break

