import os
import torch
import torch.nn as nn

import random

from crowd_model import MCNN
from dataloader import CrowdDataset


if __name__=="__main__":
    torch.backends.cudnn.enabled=False
    
    device=torch.device("cuda")
    mcnn=MCNN().to(device)
    criterion=nn.MSELoss(size_average=False).to(device)
    optimizer = torch.optim.SGD(mcnn.parameters(), lr=1e-6,
                                momentum=0.90)
    
    #Loading Files 
    img_root= "/content/drive/My Drive/ShanghaiTech/part_A/train_data/images"
    groundtruth_dmap= "/content/drive/My Drive/ShanghaiTech/part_A/train_data/ground-truth"
    dataset=CrowdDataset(img_root,groundtruth_dmap,4)
    dataloader=torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=True)

    test_img = "/content/drive/My Drive/ShanghaiTech/part_A/test_data/images"
    test_gt_dmap = "/content/drive/My Drive/ShanghaiTech/part_A/test_data/ground-truth"
    dataset_test =CrowdDataset(test_img,test_gt_dmap,4)
    test_dataloader=torch.utils.data.DataLoader(dataset_test,batch_size=1,shuffle=False)

    #Make Checkpoint directory
    if not os.path.exists('/content/drive/My Drive/Colab Notebooks/checkpoints'):
        os.mkdir('/content/drive/My Drive/Colab Notebooks/checkpoints')
    min_mae=10000
    min_epoch=-1
    train_loss_list=[]
    epoch_list=[]
    test_error_list=[]
    for epoch in range(0,100):

        mcnn.train()
        epoch_loss=0
        for i,(img,groundtruth_dmap) in enumerate(dataloader):
            img=img.to(device)
            groundtruth_dmap=groundtruth_dmap.to(device)

            #Propagation - Forward
            et_dmap=mcnn(img)

            # Loss
            loss=criterion(et_dmap,groundtruth_dmap)
            epoch_loss+=loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("epoch:",epoch,"loss:",epoch_loss/len(dataloader))
        epoch_list.append(epoch)
        train_loss_list.append(epoch_loss/len(dataloader))
                
        m_name = 'epoch_'+str(epoch)+'.pt'
        path = F"/content/drive/My Drive/Colab Notebooks/checkpoints/{m_name}"
        print(path)
        torch.save(mcnn.state_dict(), path)

        mcnn.eval()
        mae=0
        for i,(img,groundtruth_dmap) in enumerate(test_dataloader):
            img=img.to(device)
            groundtruth_dmap=groundtruth_dmap.to(device)

            et_dmap=mcnn(img)
            mae+=abs(et_dmap.data.sum()-groundtruth_dmap.data.sum()).item()
            del img,groundtruth_dmap,et_dmap
        if mae/len(test_dataloader)<min_mae:
            min_mae=mae/len(test_dataloader)
            min_epoch=epoch
        test_error_list.append(mae/len(test_dataloader))
        print("epoch:"+str(epoch)+" error:"+str(mae/len(test_dataloader))+" min_mae:"+str(min_mae)+" min_epoch:"+str(min_epoch))
        