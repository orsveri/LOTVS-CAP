import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from src.model import accident
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
from src.bert import opt
from src.dataset import DADA2K
from natsort import natsorted
import argparse

#os.environ['CUDA_VISIBLE_DEVICES']= '1'
transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

# device = ("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cuda')
num_epochs = 50
# learning_rate = 0.0001
batch_size = 1
shuffle = True
pin_memory = True
num_workers = 1
rootpath=r'/home/sorlova/data/LOTVS/DADA/DADA2000'
frame_interval=1
input_shape=[224,224]
seed = 123
np.random.seed(seed)
torch.manual_seed(seed)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
val_data=DADA2K(rootpath , 'testing', interval=1,transform=transform)
valdata_loader=DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False,
                                  num_workers=num_workers, pin_memory=True,drop_last=True)


def write_scalars(logger, epoch, loss):
    logger.add_scalars('train/loss',{'loss':loss}, epoch)

def write_test_scalars(logger, epoch, losses, metrics):
    # logger.add_scalars('test/loss',{'loss':loss}, epoch)
    logger.add_scalars('test/losses/total_loss',{'Loss': losses}, epoch)
    logger.add_scalars('test/accuracy/AP',{'AP':metrics['AP'], 'PR80':metrics['PR80']}, epoch)
    logger.add_scalars('test/accuracy/time-to-accident',{'mTTA':metrics['mTTA'], 'TTA_R80':metrics['TTA_R80']}, epoch)

def test(test_dataloader, model):
    all_pred = []
    all_labels = []
    losses_all = []
    all_toas = []
    df_clips = []
    df_frame_names = []
    df_preds = []
    df_labels = []
    df_ttcs = []
    model.eval()
    with torch.no_grad():
        loop = tqdm(test_dataloader,total = len(test_dataloader), leave = True)
        for imgs,focus,extra,labels_,texts in loop:
            # torch.cuda.empty_cache()
            assert imgs.size()[0] == 1
            info = extra["data_info"]
            num_frames = imgs.size()[1]
            imgs=imgs.to(device)
            focus=focus.to(device)
            #labels = label  # only for batch 1
            toa = info[:, 4].to(device)
            labels_ = np.array(labels_).astype(int)
            labels_ = torch.from_numpy(labels_)
            labels = torch.nn.functional.one_hot(torch.max(labels_, dim=-1)[0], num_classes=2)
            labels = labels.to(device)
            total_loss, outputs = model(imgs,focus,labels.long(),toa,texts)

            #gather results and ground truth
            outputs = [o.detach().cpu().numpy()[0] for o in outputs]
            toas = np.squeeze(toa.cpu().numpy()).astype(np.int64)
            #
            pred_frames = np.zeros((1, num_frames), dtype=np.float32)
            for t in range(num_frames):
                pred = outputs[t]
                pred_frames[0, t] = np.exp(pred[1]) / np.sum(np.exp(pred))
            label_onehot = labels.cpu().numpy()
            label = np.reshape(label_onehot[:, 1], [batch_size,])
            # gather results and ground truth
            all_pred.append(pred_frames)
            all_labels.append(label)
            all_toas.append(toas)

            losses_all.append(total_loss["total_loss"].item())

            # my file!
            start, end = info[0, 2:4]
            filenames = [f"{str(ts).zfill(4)}.png" for ts in range(start, end + 1)]
            clip_names = [f"{str(info[0, 0].item())}/{str(info[0, 1].item()).zfill(3)}"] * len(filenames)
            ttc = extra["ttc"][0].detach().cpu().numpy()
            labels_ = labels_[0].detach().cpu().numpy()
            df_preds.extend(outputs)
            df_labels.append(labels_)
            df_clips.extend(clip_names)
            df_frame_names.extend(filenames)
            df_ttcs.append(ttc)

            loop.set_postfix(val_loss = sum(losses_all))

    all_pred = np.concatenate(all_pred, axis=1)
    all_labels = np.concatenate(all_labels, axis=0)
    all_toas = np.concatenate(all_toas, axis=None)

    # my file!
    logits = np.stack(df_preds, axis=0)
    df_labels = np.concatenate(df_labels, axis=0)
    df_ttcs = np.concatenate(df_ttcs, axis=0)

    df = pd.DataFrame({
        "clip": df_clips,
        "filename": df_frame_names,
        "logits_safe": logits[:, 0],
        "logits_risk": logits[:, 1],
        "label": df_labels,
        "ttc": df_ttcs
    })

    return all_pred, all_labels, all_toas, df

def test_data():
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
    ckpt = str(opt.e).zfill(2) #"00"
    ckpt_path = f'logs/dada2k_seq96_b4_2kiter/ckpts/ckpt_{ckpt}.pth'
    save_to = f"logs/dada2k_seq96_b4_2kiter/preds_dada/predictions_{ckpt}.csv"
    weight = torch.load(ckpt_path)["model_state_dict"]
    model=accident(h_dim,n_layers,depth,adim,heads,num_tokens,c_dim,s_dim1,s_dim2,keral,num_class).to(device)
    # Only for shared by authors
    #weight["fusion.image_model.downsampling.weight"] = weight.pop("fusion.image_model.upsampling.weight")
    #weight["fusion.image_model.downsampling.bias"] = weight.pop("fusion.image_model.upsampling.bias")
    model.load_state_dict(weight, strict=True)  # hmmm...
    model.eval()
    print('------Starting evaluation------')
    os.makedirs(os.path.dirname(save_to), exist_ok=True)
    all_pred, all_labels, all_toas, df = test(valdata_loader,model)
    df.to_csv(save_to, index=True, header=True)


if __name__=="__main__":
    test_data()