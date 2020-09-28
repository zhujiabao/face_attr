import h5py
import numpy as np

path = "../data/LFWA/indices_train_test.mat"
data = h5py.File(path)
#print(data['keys'])
# mat = np.transpose(data)  #['indices_identity_test' 'indices_identity_train' 'indices_img_test' 'indices_img_train']
# print(data['indices_identity_test'])
# indices_identity_test = np.transpose(data['indices_identity_test'])
# print(indices_identity_test)
data = data['indices_identity_test'][:]
print(data)

path2 = "../data/LFWA/lfw_att_40.mat"
attr = h5py.File(path2) #['#refs#' 'AttrName' 'label' 'name']
print(attr.keys())
#mat = np.transpose(attr)
array = attr['name']
for i in range(array.shape[1]):
    name = ''.join([chr(v[0]) for v in data[(array[0][i])]])
    print(name)

