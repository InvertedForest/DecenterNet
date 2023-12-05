import numpy as np
import torch
import numbers
import torchvision
import cv2
import matplotlib.pyplot as plt
ttt = torchvision.transforms.ToPILImage()
from visual import *

cmp='rainbow'
interpolation=None
def show_img(*args):
    mean=(0.485, 0.456, 0.406)
    std=(0.229, 0.224, 0.225)

    n = len(args)
    for i in range(len(args)):
        plt.subplot(n, 1, i+1)
        if args[i].size(0) == 3:
            img = args[i].cpu().clone().numpy().transpose(1, 2, 0)
            img = (img * std + mean) * 255
            plt.imshow(img.astype('uint8'))
        else:
            img = args[i].cpu().clone()
            img = ttt(img)
            plt.imshow(img, cmap=cmp, interpolation=None)
        plt.savefig('test.png')
    plt.close("all")
    
def cv_img(img, st):
    mean=(0.485, 0.456, 0.406)
    std=(0.229, 0.224, 0.225)

    if img.size(0) == 3:
        img = img.cpu().clone().numpy().transpose(1, 2, 0)
        img = (img * std + mean) * 255
        cv2.imwrite(f'test_{st}.png', img[...,::-1])
    else:
        img = img.cpu().clone()
        img = img/img.max()*200
        img = np.array(img, np.uint8)
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        cv2.imwrite(f'test_{st}.png', img)

def double_img(alp, ori):
    mean=(0.485, 0.456, 0.406)
    std=(0.229, 0.224, 0.225)

    img = ori.cpu().clone().numpy().transpose(1, 2, 0)
    img = (img * std + mean) * 255
    plt.imshow(img.astype('uint8'))

    img = alp.cpu().clone()
    img = ttt(img)
    img = img.resize((512, 512))
    plt.imshow(img, alpha = 0.4, cmap=cmp, interpolation=None)

    plt.savefig('test.png')
    plt.close("all")

# n为递归深度
def pri(item, n=-1, key=0):
    st= ' .'
    n += 1
    if isinstance(item, dict):
        for k,v in item.items():
            pri(k, n, 1)
            pri(v, n)

    elif isinstance(item, (list)):
        print('list', end=' ')
        if isinstance(item[0], (numbers.Number, str)):
            print(f"\033[0;37;40m[{len(item)}]\033[0m", end=' ')
        else:
            for v in item:
                pri(v, n)

    elif isinstance(item, (np.ndarray)):
        print('np.ndarray', end=' ')
        print(f"\033[0;37;40m[{item.shape}]\033[0m", end=' ')

    elif isinstance(item, torch.Tensor):
        print('tensor', end=' ')
        print(f"\033[0;37;40m{str(item.shape)[11:-1]}\033[0m", end=' ')

    elif isinstance(item, (numbers.Number)):
        print(f"\033[0;37;40m{item}\033[0m", end=' ')

    elif isinstance(item, (str)):
        if key == 0:
            print(f"\033[0;37;40m{item}\033[0m", end=' ')
        else:
            print('\n' + st*n + f"\033[0;35;40m{item}: \033[0m", end='')

    elif isinstance(item, (tuple)):
        print('tuple', end=' ')
        for i in item:
            pri(i,n)
    else:
        print("str({type(item)})", end=' ')
    
    print('\n')

# print('import visual ok')

'''
a = torch.tensor(final_poses[0][:,:2])
tmp = torch.zeros(427,640)
loc = a.long()
tmp[loc[..., 1], loc[..., 0]] = 1
double_img(tmp,image_resized[0])
show_img(tmp)
'''

'''
tmp = torch.zeros_like(heatmap_avg[0,0]).cpu()
person = (posemap[0,:,201//4,242//4].clone().detach().cpu() + 0.5).to(torch.long)
tmp[person[:,1], person[:,0]]=1
cv_img(tmp,'tmp1')

tmp = torch.zeros_like(heatmap_avg[0,0]).cpu()
person = (posemap[0,:,64,61].clone().detach().cpu()).to(torch.long)
w = tmp.size(1)
h = tmp.size(0)
tmp[person[:,1]%h, person[:,0]%w]=1
cv_img(tmp,'tmp1')
'''
# numpy
'''
a = poses[0][...,:-1]
tmp = torch.zeros_like(heatmap_avg[0,0]).cpu()
w = tmp.size(1)
h = tmp.size(0)
person = a[1]
tmp[person[:,1]%h, person[:,0]%w]=1
cv_img(tmp,'tmp1')
'''
# torch
'''
tmp = torch.zeros_like(heatmap_avg[0,0]).cpu()
person = (poses[0]).long()
tmp[person[:,1], person[:,0]]=1
cv_img(tmp,'tmp1')
'''

'''
cv_img(image_resized[0],'ori')
cv_img(posemap[0].sum(-1).sum(0), 'tmp2')
'''

'''
cv_img(nms_heatmaps[0].sum(0), "nms")

cv_img(heatmap_avg[0].sum(0), "heat")
'''
'''
cv_img(offsets[0][10], "off10")

'''