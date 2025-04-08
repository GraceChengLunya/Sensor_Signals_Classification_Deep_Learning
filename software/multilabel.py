import torch
import numpy as np
from src.models import U_Net
from src.handler import TrainValHandler,TestHandler
from matplotlib import pyplot as plt
import argparse

def arg_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m','--Mode',type=str,metavar='',default='train',
                        help='specify training or testing')
    parser.add_argument('-ts','--TrainSet',type=str,metavar='',default='/media/gaofeng/Extreme SSD/datasets/spectrum_sensing_stitching/processed/train.h5',
                        help='filepath of training set')
    parser.add_argument('-vs','--ValSet',type=str,metavar='',default='/media/gaofeng/Extreme SSD/datasets/spectrum_sensing_stitching/processed/test.h5',
                        help='filepath of valiation set')
    parser.add_argument('-d','--Device',type=int,default=-1,metavar='',
                        help='specify the gpu device, -1 means cpu')
    return parser.parse_args()

def train(args):
    trainset = args.TrainSet
    print("trainset:", trainset)
    valset = args.ValSet
    model = U_Net(2,6,is_attention=True,alpha=1,beta=5) # change for unknown
    
    if args.Device == -1:
        device = 'cuda:0'
    else:
        device = f'cuda:{args.Device}'

    model = model.to(device)


    batchsize = 256
    lr=1e-3
    epochs=100
    pt=30
    ckpt = 'multilabel.pth'
    handler = TrainValHandler(model,device,trainset,valset,ckpt,lr=lr,epochs=epochs,patience=pt,batchsize=batchsize)
    history = handler.train()

    '''model = U_Net(2,6,is_attention=True,alpha=1,beta=5)
    device = 'cuda:0'
    model = model.to(device)
    ckpt = 'multilabel.pth'''


    # retrieve the best model and convert to onnx
    model.load_state_dict(torch.load(ckpt), strict=False)
    model.eval()
    dummy_input = torch.randn(1, 2, 1024, requires_grad=True)
    dummy_input = dummy_input.to(device)
    torch_out = model(dummy_input)
    torch.onnx.export(model,         # model being run 
            dummy_input,       # model input (or a tuple for multiple inputs) 
            "multilabel.onnx",       # where to save the model
            export_params=True,  # store the trained parameter weights inside the model file 
            opset_version=11,    # the ONNX version to export the model to 
            do_constant_folding=True,  # whether to execute constant folding for optimization 
            input_names = ['modelInput'],   # the model's input names 
            output_names = ['modelOutput'], # the model's output names 
            dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes 
                                'modelOutput' : {0 : 'batch_size'}}) 
    print(" ") 
    print('Model has been converted to ONNX') 

def test(args):
    testset = args.ValSet
    trainset = args.TrainSet
    model = U_Net(2,6,is_attention=True,alpha=1,beta=5)  # change for unknown
    if args.Device == -1:
        device = 'cuda:0'
    else:                       #DKASFJDKJASFKLDJASFKLDSJFKLJDSKLFJDSFJDASJFDJSFKLDSAKFKDS
        device = 'cuda:0' 
    batchsize = 256
    ckpt = 'multilabel_1205_Wifi_30.pth'
    model.load_state_dict(torch.load(ckpt), strict=False)
    handler = TestHandler(model,device,testset,batchsize)
    iou, recall, precision = handler.evaluate()
    print('validation results:')
    print('iou:',iou)
    print('precision:',precision)
    print('recall:',recall)

    train_handler = TestHandler(model,device,trainset,batchsize)
    iou, recall, precision = train_handler.evaluate()
    print('train results:')
    print('iou:',iou)
    print('precision:',precision)
    print('recall:',recall)
    
if __name__ == '__main__':
    args = arg_parser()
    if args.Mode == 'train':
        train(args)
    else:
        test(args)
