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

# test function
def test(test_dataloader,test_path,thr):
    temp = []
    model_map = {0:'model_acc', 1:'model_micro_f1', 2:'model_macro_f1'}
    device = 'cuda:0'
    df = pd.read_excel(test_path+"save.xlsx")
    for r in range(3):
        model_name = model_map[r]
        model = torch.load(test_path+"%s"%model_name+'.pt')
        model.to(device)
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(test_dataloader):
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
            print("%s, overall: Micro-F1 %.4f, Macro-F1 %.4f, Accuracy %.4f"%(model_name,micro_f1,macro_f1,acc))
            print("%s, level 1: Micro-F1 %.4f, Macro-F1 %.4f, Accuracy %.4f"%(model_name,micro_f11,macro_f11,acc1))
            print("%s, level 2: Micro-F1 %.4f, Macro-F1 %.4f, Accuracy %.4f"%(model_name,micro_f12,macro_f12,acc2))
            print("%s, level 3: Micro-F1 %.4f, Macro-F1 %.4f, Accuracy %.4f"%(model_name,micro_f13,macro_f13,acc3))
            print('\n')
            acc = acc.cpu().numpy()
            acc1 = acc1.cpu().numpy()
            acc2 = acc2.cpu().numpy()
            acc3 = acc3.cpu().numpy()
            temp.append(['test_'+"%s"%model_name,micro_f1,macro_f1,acc,micro_f11,macro_f11,acc1,micro_f12,macro_f12,acc2,micro_f13,macro_f13,acc3])
            df.loc['test_'+"%s"%model_name] = temp[r]

    df.loc['test_'+'best_ave'] = ['test_'+'average']+[(a+b+c)/3 for a,b,c in zip(temp[0][1:],temp[1][1:],temp[2][1:])]
    df.to_excel(test_path+"save.xlsx",index=False)

