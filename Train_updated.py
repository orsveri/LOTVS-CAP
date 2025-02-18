import os
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from src.model import accident_frame
from src.bert import opt
from src.dada2k_dataset import FrameClsDataset_DADA
from src.data_utils import ShortSampler


os.environ['CUDA_VISIBLE_DEVICES']= '0'
num_workers = 4
seed = 123
np.random.seed(seed)
torch.manual_seed(seed)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

num_epochs = 20
batch_size = 4
nb_samples_per_epoch = 10000
shuffle = True
pin_memory = True
rootpath=r'/mnt/experiments/sorlova/datasets/LOTVS/DADA/DADA2000'
#frame_interval=1
seq_length = 96
seq_step = 3
input_shape = [224, 224]

train_data = FrameClsDataset_DADA(
    anno_path="DADA2K_my_split/training.txt",
    data_path=rootpath,
    mode='train',
    view_len=seq_length,
    view_step=seq_step,
    orig_fps=30,
    target_fps=30,
    keep_aspect_ratio=True,
    crop_size=224,
    short_side_size=320,
    loss_name="crossentropy"
)

sampler_train = ShortSampler(
    train_data, num_samples_per_epoch=nb_samples_per_epoch, shuffle=True
)

traindata_loader=DataLoader(
    dataset=train_data, batch_size=batch_size, sampler=sampler_train,
    num_workers=num_workers, pin_memory=True, drop_last=True
)

# val_data = FrameClsDataset_DADA(
#     anno_path="DADA2K_my_split/validation.txt",
#     data_path=rootpath,
#     mode='validation',
#     view_len=seq_length,
#     view_step=seq_step,
#     orig_fps=30,
#     target_fps=30,
#     keep_aspect_ratio=True,
#     crop_size=224,
#     short_side_size=320,
#     loss_name="crossentropy"
# )

# valdata_loader=DataLoader(dataset=val_data, batch_size=batch_size , shuffle=False,
#                                   num_workers=num_workers, pin_memory=True,drop_last=True)


def write_scalars(logger, epoch, loss):
    logger.add_scalars('train',{'loss':loss}, epoch)

def write_test_scalars(logger, epoch, losses, metrics):
    # logger.add_scalars('test/loss',{'loss':loss}, epoch)
    logger.add_scalars('test/total_loss',{'Loss': losses}, epoch)
    logger.add_scalars('test/AP',{'AP':metrics['AP'], 'PR80':metrics['PR80']}, epoch)
    logger.add_scalars('test/time-to-accident',{'mTTA':metrics['mTTA'], 'TTA_R80':metrics['TTA_R80']}, epoch)

def train():
    logs_dir = 'logs/dada2k_seq96_b4_10kperep'
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    model_dir = os.path.join(logs_dir, "ckpts")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    logger = SummaryWriter(logs_dir)
    h_dim = 256
    n_layers = 1
    depth=4
    adim=opt.adim
    heads=opt.heads
    num_tokens=opt.num_tokens
    c_dim=opt.c_dim
    s_dim1=opt.s_dim1
    s_dim2=opt.s_dim2
    keral=opt.keral
    num_class=opt.num_class
    # model = AccidentXai(num_classes, x_dim, h_dim, z_dim,n_layers).to(device)
    model=accident_frame(h_dim,n_layers,depth,adim,heads,num_tokens,c_dim,s_dim1,s_dim2,keral,num_class).to(device)

    # opt1=torch.optim.Adam([
    #     {'params': model.fusion.parameters(),'lr':1e-3},
    #     {'params': model.features.parameters(), 'lr': 1e-3},
    #     {'params': model.deconv.parameters(), 'lr': 1e-3},
    # ])
    # opt2=torch.optim.Adam([
    #     {'params': model.fusion.parameters(),'lr':1e-2},
    #     {'params': model.features.parameters(), 'lr': 1e-3},
    #     {'params': model.gru_net.parameters(), 'lr': 1e-2}
    # ])
    # opt1 = torch.optim.Adam([
    #     {'params': model.fusion.parameters(), 'lr': 1e-6},
    #     {'params': model.features.parameters(), 'lr': 1e-5},
    #     {'params': model.deconv.parameters(), 'lr':1e-5}
    # ])
    #
    # opt2 = torch.optim.Adam([
    #     {'params': model.fusion.parameters(), 'lr':1e-5},
    #     {'params': model.features.parameters(), 'lr': 1e-5},
    #     {'params': model.gru_net.parameters(), 'lr': 1e-5},
    # ])

    opt1 = torch.optim.Adam([
        {'params': model.fusion.parameters(), 'lr': 1e-6},
        {'params': model.features.parameters(), 'lr': 1e-6},
        {'params': model.deconv.parameters(), 'lr': 1e-4},
        {'params': model.gru_net.parameters(), 'lr': 1e-5},
    ])

    # for name, param in model.gru_net.named_parameters():
    #     if 'gru.weight' in name or 'gru.bias' in name:
    #         param.requires_grad = True
    #     elif 'dense1' in name or 'dense2' in name:
    #         param.requires_grad = True
    #     else:
    #         param.requires_grad = False

    scheduler1 = torch.optim.lr_scheduler.StepLR(opt1, step_size=1, gamma=0.1)

    model.train()
    for epoch in range(num_epochs):
        loop = tqdm(traindata_loader ,total = len(traindata_loader), leave = True)
        sampler_train.set_epoch(epoch)
        for imgs, focus, labels, info, texts in loop:
            loop.set_description(f"Epoch  [{epoch}/{num_epochs}]")
            # print(imgs.shape)
            imgs = imgs.to(device)
            focus = focus.to(device)
            labels = np.array(labels).astype(int)
            labels = torch.from_numpy(labels)
            labels = labels.to(device)
            model.to(device)
            loss, outputs = model(imgs, focus, labels.long(), texts)
            opt1.zero_grad()
            loss['total_loss'].mean().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),10)
            opt1.step()
            loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
            loop.set_postfix(loss = loss['total_loss'].item())
        write_scalars(logger,epoch,loss['total_loss'])

        if (epoch+1) % 5 == 0:
            scheduler1.step()
        #test and evaluate the model
        if (epoch+1) % 1==0:
            model.train()
            # save model
            # best_model_file = os.path.join(model_dir, 'best_model.pth')
            model_file = os.path.join(model_dir, 'ckpt_%02d.pth'%(epoch))
            torch.save(model.state_dict(), model_file)
        logger.close()

if __name__ == "__main__":
    train()
