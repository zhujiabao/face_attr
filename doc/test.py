import json

class predict_decoder(object):
    def __init__(self, dataset="CelebA"):
        with open('./label.json', 'r') as f:
            self.label_list = json.load(f)[dataset]
        with open('./attribute.json', 'r') as f:
            self.attribute_dict = json.load(f)[dataset]
        self.dataset = dataset
        self.num_label = len(self.label_list)

    def decode(self, pred):
        #pred = pred.squeeze(dim=0)
        for idx in range(self.num_label):
            name, chooce = self.label_list[idx], self.attribute_dict[self.label_list[idx]][pred[idx]]
            #if chooce
            print('{}: {}'.format(name, chooce))
            #return name, chooce


if __name__ == '__main__':
    dec = predict_decoder(dataset="CelebA")
    dec.decode(pred=[1,0,1,0,0,1,0,1,0,0,1,0,1,0,0,1,0,1,0,0,1,0,1,0,0,1,0,1,0,0,1,0,1,0,0,1,0,1,0,0])