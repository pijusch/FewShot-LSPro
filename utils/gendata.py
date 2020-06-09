import pandas as pd
import pickle

class Data:
    def __init__(self,rel_list,train_num,test_num):
        self.rel_list = rel_list
        self.test_num = test_num
        self.train_num = train_num
        csv = pickle.load(open("../data/all_analogies.pkl","rb"))
        # csv = pd.read_csv('all_analogies.csv',header=None)
        self.data = csv[csv['relation'].isin(rel_list)]
    
    def get_train_set(self):
        train = []
        for i in self.rel_list:
            train.append(self.data[self.data['relation']==i].sample(n=self.train_num))
        return train

    def get_test_set(self):
        test = []
        for i in self.rel_list:
            test.append(self.data[self.data['relation']==i].sample(n=self.test_num))
        return test

if __name__ == "__main__":
    data = Data(["01-all-capital-cities.txt"],5,100)
    print(data.get_test_set())