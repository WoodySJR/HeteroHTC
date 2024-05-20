import torch, pandas as pd, numpy as np, math
from torch import nn
#from torch.nn.utils import clip_grad_norm_ as clip
from torch.nn.functional import binary_cross_entropy as BCE
from torch.optim import Adam, SGD, AdamW
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
import openpyxl

# training function (including early stopping)
def train(model, train_dataloader, val_dataloader, num_epochs, lr, budget, save_path, thr):
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.0000001)
    acc_max = -math.inf
    micro_f1_max = -math.inf
    macro_f1_max = -math.inf
    #count_e_acc = 0
    count_e_mi = 0
    count_e_ma = 0
    epoch_best_acc = 0
    epoch_best_mi = 0
    epoch_best_ma = 0
    save = np.zeros([num_epochs+8,13])
    for e in range(num_epochs):
        loss_value = 0
        acc_value = 0
        count = 0
        with tqdm(total=len(train_dataloader), leave=False) as t: 
            for inputs, labels in train_dataloader:
                labels = labels[:,0:130]
                t.set_description("Epoch "+str(e))
                optimizer.zero_grad()
                prob = model(inputs)
                pred = (prob>thr).detach()
                acc = torch.sum(torch.all(pred==labels, dim=1))/inputs.shape[0]
                acc_value += acc
                losses = BCE(prob, labels.float(), reduction="none")
                #losses = BCE(prob, labels.float(), reduction="none", 
                #             weight=torch.cat((labels[:,0:20]*9+1, labels[:,20:100]*29+1, labels[:,100:225]*59+1), dim=1))
                loss = torch.sum(losses)/losses.shape[0]
                loss_value += loss.detach().item()
                count += 1
                loss.backward()
                #clip(model.parameters(), max_norm=20, norm_type=2)
                optimizer.step()
                t.set_postfix(loss=loss_value/count, acc=acc_value/count)
                t.update(1)
            with torch.no_grad():
                for i, (inputs, labels) in enumerate(val_dataloader):
                    labels = labels[:,0:130]
                    prob = model(inputs)
                    pred = (prob>thr).float()
                    if i==0:
                        preds = pred
                        targets = labels
                    else:
                        preds = torch.cat((preds,pred), dim=0)
                        targets = torch.cat((targets,labels), dim=0)
                micro_f1 = f1_score(targets.cpu(), preds.cpu(), average="micro")
                macro_f1 = f1_score(targets.cpu(), preds.cpu(), average="macro")
                micro_f11 = f1_score(targets[:,0:7].cpu(), preds[:,0:7].cpu(), average="micro")
                macro_f11 = f1_score(targets[:,0:7].cpu(), preds[:,0:7].cpu(), average="macro")
                micro_f12 = f1_score(targets[:,7:53].cpu(), preds[:,7:53].cpu(), average="micro")
                macro_f12 = f1_score(targets[:,7:53].cpu(), preds[:,7:53].cpu(), average="macro")
                micro_f13 = f1_score(targets[:,53:130].cpu(), preds[:,53:130].cpu(), average="micro")
                macro_f13 = f1_score(targets[:,53:130].cpu(), preds[:,53:130].cpu(), average="macro")
                acc = torch.sum(torch.all(preds==targets, dim=1))/targets.shape[0]
                acc1 = torch.sum(torch.all(preds[:,0:7]==targets[:,0:7], dim=1))/targets.shape[0]
                acc2 = torch.sum(torch.all(preds[:,7:53]==targets[:,7:53], dim=1))/targets.shape[0]
                acc3 = torch.sum(torch.all(preds[:,53:130]==targets[:,53:130], dim=1))/targets.shape[0]
                print("Epoch %d overall: Micro-F1 %.4f, Macro-F1 %.4f, Accuracy %.4f"%(e,micro_f1,macro_f1,acc))
                print("Epoch %d level 1: Micro-F1 %.4f, Macro-F1 %.4f, Accuracy %.4f"%(e,micro_f11,macro_f11,acc1))
                print("Epoch %d level 2: Micro-F1 %.4f, Macro-F1 %.4f, Accuracy %.4f"%(e,micro_f12,macro_f12,acc2))
                print("Epoch %d level 3: Micro-F1 %.4f, Macro-F1 %.4f, Accuracy %.4f"%(e,micro_f13,macro_f13,acc3))
                acc = acc.cpu().numpy()
                acc1 = acc1.cpu().numpy()
                acc2 = acc2.cpu().numpy()
                acc3 = acc3.cpu().numpy()
                save[e] = [e,micro_f1,macro_f1,acc,micro_f11,macro_f11,acc1,micro_f12,macro_f12,acc2,micro_f13,macro_f13,acc3]
                df = pd.DataFrame(save)
                df.columns = ['epoch','micro_f1','macro_f1','acc','micro_f11','macro_f11','acc1','micro_f12','macro_f12','acc2','micro_f13','macro_f13','acc3']

                if (acc>acc_max):
                    torch.save(model, save_path+"/model_acc.pt")
                    print("Saving model_acc parameters. ")
                    acc_max = acc
                #    count_e_acc = 0
                    epoch_best_acc = e+1
                    temp_best_acc = ['final_model_val_acc(epoch%d)'%epoch_best_acc,micro_f1,macro_f1,acc,micro_f11,macro_f11,acc1,micro_f12,macro_f12,acc2,micro_f13,macro_f13,acc3]
                #else:
                #    count_e_acc += 1
                if (micro_f1>micro_f1_max):
                    torch.save(model, save_path+"/model_micro_f1.pt")
                    print("Saving model_micro_f1 parameters. ")
                    micro_f1_max = micro_f1
                    count_e_mi = 0
                    epoch_best_mi = e+1
                    temp_best_mi = ['final_model_val_micro_f1(epoch%d)'%epoch_best_mi,micro_f1,macro_f1,acc,micro_f11,macro_f11,acc1,micro_f12,macro_f12,acc2,micro_f13,macro_f13,acc3]
                else:
                    count_e_mi +=1
                if (macro_f1>macro_f1_max):
                    torch.save(model, save_path+"/model_macro_f1.pt")
                    print("Saving model_macro_f1 parameters. ")
                    macro_f1_max = macro_f1
                    count_e_ma = 0
                    epoch_best_ma = e+1
                    temp_best_ma = ['final_model_val_macro_f1(epoch%d)'%epoch_best_ma,micro_f1,macro_f1,acc,micro_f11,macro_f11,acc1,micro_f12,macro_f12,acc2,micro_f13,macro_f13,acc3]
                else:
                    count_e_ma +=1

                if (count_e_mi>budget)&(count_e_ma>budget): # no improvement for more than budget epochs
                    print("From perspective of all the three evaluation metrics: acc, micro f1 and macro f1..."+"\n"
                          " Early stopped after "+str(e)+" epochs."+"\n"
                          " Best model_acc saved at epoch "+str(epoch_best_acc)+". "+"\n"
                          " Best model_micro_f1 saved at epoch "+str(epoch_best_mi)+". "+"\n"
                          " Best model_macro_f1 saved at epoch "+str(epoch_best_ma)+". ")
                    break
                
                df.loc['final_acc'] = temp_best_acc
                df.loc['final_mi'] = temp_best_mi
                df.loc['final_ma'] = temp_best_ma
                df.loc['best_ave'] = ['average']+[(a+b+c)/3 for a,b,c in zip(temp_best_acc[1:],temp_best_mi[1:],temp_best_ma[1:])]
                df.to_excel(save_path+"/save.xlsx",index=False)
