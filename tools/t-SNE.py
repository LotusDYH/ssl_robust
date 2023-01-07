import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

## scale to [0,1]
def plot_embedding(data):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    return data


def t_sne(n_components, data_path, output_path):
    data = np.load(data_path)
    data_1 = data['a']
    data_2 = data['b']
    data_3 = data['c']
    data = np.vstack((data_1, data_2))
    target = [i for i in range(5)]
    target.extend(target)
    tsne_digits = TSNE(n_components=n_components).fit_transform(data)
    aim_data = plot_embedding(tsne_digits)
    plt.figure()
    plt.subplot(111)
    plt.scatter(np.vstack((aim_data[5:10, 0], aim_data[len(data)//2+5:len(data)//2+10, 0])), \
                np.vstack((aim_data[5:10, 1], aim_data[len(data)//2+5:len(data)//2+10, 1])), c=target)
    plt.title("T-SNE")
    plt.savefig(output_path)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    data_args = parser.add_argument_group('t-SNE')
    data_args.add_argument('--data_path', type=str)
    data_args.add_argument('--output', type=str)

    args = parser.parse_args()
    t_sne(2, args.data_path, args.output)