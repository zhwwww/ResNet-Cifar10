import pickle
from PIL import Image
import os

class Cifar():
    def __init__(self, batchpath, root):
        self.batchpath = batchpath
        self.root = root
        self.img_label = []

    def make_file_list(self, filename):
        with open(filename, 'a+') as f:
            for (imgname, label) in self.img_label:
                f.writelines("{}\t{}\n".format(imgname, label))

    def data_parse(self):
        with open(self.batchpath, 'rb') as fb:
            cifar_dict = pickle.load(fb, encoding='latin1')
            batch_label, labels, data, filenames = cifar_dict['batch_label'], cifar_dict['labels'], \
                                                   cifar_dict['data'], cifar_dict['filenames']
            for i, (img, label, filename) in enumerate(zip(data, labels, filenames)):
                img = img.reshape(3, 32, 32)
                r = Image.fromarray(img[0]).convert('L')
                g = Image.fromarray(img[1]).convert('L')
                b = Image.fromarray(img[2]).convert('L')
                img = Image.merge('RGB', (r, g, b))

                self._save_img(self.root, img, label, filename)

    def _save_img(self, root, img, label, filename):
        save_path = os.path.join(root, str(label))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_name = os.path.join(save_path, filename)
        self.img_label.append((save_name, label))
        print(save_name)
        img.save(save_name)


if __name__ == "__main__":
    for i in range(5):
        cifar = Cifar("./cifar-10-batches-py/data_batch_{}".format(i+1), "train")
        cifar.data_parse()
    cifar = Cifar("./cifar-10-batches-py/test_batch", "test")
    cifar.data_parse()
    #cifar.make_file_list('data_{}.txt'.format(i+1))
    #with open("./cifar-10-batches-py/batches.meta", 'rb') as bm:
    #    data = pickle.load(bm, encoding='bytes')
    #    print(data.keys())
    #    print(data[b'num_cases_per_batch'])
    #    print(data[b'label_names'])
    #    print(data[b'num_vis'])
