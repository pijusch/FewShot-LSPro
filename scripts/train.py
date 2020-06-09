import torch
import numpy as np 
from model import LSPro
from gendata import Data



class FOMAML:
    def __init__(self):
        self.teacher_lr = 0.0002
        self.student_lr = 0.002
        self.alpha1 = 0#1e-1
        self.alpha2 = 1e1
        self.alpha3 = 1e1
        self.alpha4 = 1e1
        self.alpha5 = 0#-1e0
        self.alpha6 = 0#-1e3
        self.alpha7 = 0#-1e3
        self.alpha8 = 1e1
        self.alpha9 = 1e1
        return

    def make_tensor(self,df):
        names = [list(df['head']),list(df['tail'])]
        embeds = [torch.Tensor(list(df['heade'])),torch.Tensor(list(df['taile']))]
        return names,embeds


    def main_trainer(self,teacher_model,num_tasks, iters):
        teacher_optim = torch.optim.Adam(teacher_model.parameters(), lr=self.teacher_lr, betas=(0,0.9))

        df = data.get_train_set()

        for task in range(num_tasks):

            student_model = LSPro(300,weights=teacher_model.state_dict())
            student_optim = torch.optim.SGD(student_model.parameters(),lr=self.student_lr)

            for i in range(iters-1):
                student_model = self.student_train_step(student_model, task, student_optim, df)


            names, embeds = self.make_tensor(df[task])
            vectors = student_model.forward(embeds[0],embeds[1])
            loss = self.loss_function(model,vectors)
            final_grad = torch.autograd.grad(loss, student_model.parameters())

            for param, grad in zip(teacher_model.parameters(), final_grad):
                param.grad = grad
            teacher_optim.step()

        return model


    def student_train_step(self,model, task, optimizer,df):
        names, embeds = self.make_tensor(df[task])
        vectors = model.forward(embeds[0],embeds[1])
        loss = self.loss_function(model,vectors)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return model
        

    def fine_tune(self,model, task, iters):
        df = data.get_test_set()
        new_model = LSPro(300,weights=model.state_dict())
        new_optim = torch.optim.SGD(new_model.parameters(), lr=self.student_lr)
        for _ in range(iters-1):
            new_model = self.student_train_step(new_model, task, new_optim, df)
        return new_model


    

    def loss_function(self,model,V):
        X = torch.sqrt(torch.sum((V[0]-V[1])**2,axis=1))
        Y = torch.sqrt(torch.sum((V[2]-V[3])**2,axis=1))
        dis = []
        interdis = 0
        actual_dis = 0
        mean1 = torch.mean(V[4],axis=0)
        mean2 = torch.mean(V[5],axis=0)
        mean1_ = V[4]
        mean2_ = V[5]
        dis = torch.Tensor(dis)
        dis = torch.mean(torch.sum((V[4]-mean1)**2,axis=1))+torch.mean(torch.sum((V[5]-mean2)**2,axis=1))#- torch.mean(torch.sum((V[4]-mean2)**2,axis=1)) -torch.mean(torch.sum((V[5]-mean1)**2,axis=1))
        pairwise = V[4]-V[5]
        norm = torch.sqrt(torch.sum(pairwise**2,axis=1)).unsqueeze(1)
        pairwise =  torch.FloatTensor([1])-torch.mm(pairwise,torch.t(pairwise))/torch.mm(norm,torch.t(norm))
        pairwise = torch.mean(pairwise)
        print(self.alpha2*torch.mean((X-Y)**2),self.alpha3*torch.mean((V[2]-V[0])**2),self.alpha3*torch.mean((V[1]-V[3])**2) ,self.alpha4*pairwise,self.alpha8*dis,self.alpha9*(1-torch.sum((mean1-mean2)**2))**2)
        return self.alpha2*torch.mean((X-Y)**2)+self.alpha3*torch.mean((V[2]-V[0])**2)+self.alpha3*torch.mean((V[1]-V[3])**2) + self.alpha4*pairwise + self.alpha5*torch.sum((torch.mean(V[2],dim=1)-torch.mean(V[3],dim=1))**2)+self.alpha8*dis + self.alpha9*torch.mean(1-torch.sum((mean1_-mean2_)**2,axis=1))**2#alpha9*(1-torch.sum((mean1-mean2)**2))**2# + alpha6*torch.mean(dis) +alpha7*torch.mean(interdis) + alpha9*torch.mean(10-torch.sum((mean1_-mean2_)**2,axis=1))**2#



if __name__ == "__main__":
    relations = ["11-plural-nouns.txt","04-man-woman.txt"]
    data = Data(relations,5,10)
    fomaml = FOMAML()
    print(data)
    print(data.get_test_set())
    model = LSPro(300)
    model = fomaml.main_trainer(model,len(relations),100)
    print()
    torch.save(model,'fomaml.model')
    fomaml.fine_tune(model,1,100)