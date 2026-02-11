import torch
from tqdm import tqdm
import os

def test_classification(net,test_loader,max_iteration=None, description=None):
    pos = 0
    tot = 0
    i = 0
    max_iteration = len(test_loader) if max_iteration is None else max_iteration
    with torch.no_grad():
        q=tqdm(test_loader, desc=description)
        for inp,target in q:
            if i > max_iteration-2:
                break
            if inp.size(0) != 32: break
            inp=inp.cuda()
            target=target.cuda()
            out=net(inp)
            pos_num=torch.sum(out.argmax(1)==target).item()
            pos+=pos_num
            tot+=inp.size(0)
            q.set_postfix({"acc":pos/tot})
            i += 1
            
    print(pos/tot)
    return pos/tot