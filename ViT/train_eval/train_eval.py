import torch
import torch.nn as nn

import time
import os
import sys
import platform

root_dir = os.path.dirname(os.path.realpath("main.py"))
sys.path.insert(0, root_dir)

from utils.utils import logger_vit, get_one_hot_encoding

if platform.system() == 'Windows':
    MODEL_PATH = 'D:/.data/models_pth/vit_reproduce.pt'
else: # ubuntu
    MODEL_PATH = '/home/jw/vscode/model_pth/vit_reproduce.pt'


logger = logger_vit()


def train_vit(model,
              optimizer,
              lr_scheduler,
              loss_ft,
              train_loader,
              n_classes,
              epoch, 
              batch_size,
              continue_train=False,
              fine_tuning=False,
              name_of_cls_fc='fc',
              name_of_cls_norm='norm',
              device='cpu'):

    # for training more from previous result
    if continue_train == True:
        if os.path.isfile(MODEL_PATH):
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    # for fine tuning
    if fine_tuning == True:
        for name, params in model.named_parameters():
            if name == name_of_cls_fc+'.weight'\
                or name == name_of_cls_fc+'.bias':
                #or name == name_of_cls_norm+'.weight'\
                #or name == name_of_cls_norm+'.bias':
                params.requires_grad = True
            else:
                params.requires_grad = False # frozen pre-trained parameters
            #print(name, params.requires_grad)

    model.train()

    print('training vit is started !')
    logger.write_log('training vit is started !')
    train_start = time.time()
    for epc in range(epoch+1):
        for batch_idx, (img, label) in enumerate(train_loader):
            img = img.to(device)
            label = label.to(device)

            prediction = model(img)
            criterion = loss_ft(prediction, label)
            
            # back-propagation
            optimizer.zero_grad()
            criterion.backward()
            optimizer.step()

            if (batch_idx*batch_size) % 10000 == 0:
                print(f'[{time.strftime("%c", time.localtime())}] epoch:{epc}, batch:{batch_idx*batch_size} loss:{criterion.item()}')
                logger.write_log(f'[{time.strftime("%c", time.localtime())}] epoch:{epc}, batch:{batch_idx*batch_size} loss:{criterion.item()}')

        #print(f'[{time.strftime("%c", time.localtime())}] epoch:{epc}, loss:{criterion.item()}')
        lr_scheduler.step()
    torch.save(model.state_dict(), MODEL_PATH)
    print('total training time: {0}'.format(time.time()-train_start))
    logger.write_log('total training time: {0}'.format(time.time()-train_start))
    
def eval_vit(model,
             test_loader,
             device='cpu'):

    print('evaluation vit is start !')
    logger.write_log('evaluation vit is start !')
    eval_start = time.time()
    true_cnt = 0
    total_cnt = 0
    # load learned model
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    with torch.no_grad():
        for batch_idx, (img, label) in enumerate(test_loader):
            img = img.to(device)
            label = label.to(device)

            prediction = model(img)
            true_cnt += (torch.argmax(prediction, dim=1) == label).sum()
            total_cnt += len(label)
    
    print('evaluation time: {0}'.format(time.time()-eval_start))
    print('accuracy: {0}, true: {1}, total: {2}'.format(round((true_cnt/total_cnt).item()*100.0, 2), true_cnt, total_cnt))
    logger.write_log('evaluation time: {0}'.format(time.time()-eval_start))
    logger.write_log('accuracy: {0}, true: {1}, total: {2}'.format(true_cnt/total_cnt, true_cnt, total_cnt))

