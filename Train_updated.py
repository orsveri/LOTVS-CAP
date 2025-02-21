import os
import gc
import numpy as np
import torch
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from natsort import natsorted

from src.model import accident_frame
from src.bert import opt
from src.dada2k_dataset import FrameClsDataset_DADA
from src.data_utils import ShortSampler
import utils

cudnn.enabled = True
torch.backends.cudnn.benchmark = True


os.environ['CUDA_VISIBLE_DEVICES']= '0'
num_workers = 4
seed = 123
np.random.seed(seed)
torch.manual_seed(seed)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

num_epochs = 20
batch_size = 3
nb_batches_per_epoch = 1600
shuffle = True
pin_memory = True
rootpath=r'/home/sorlova/data/LOTVS/DADA/DADA2000'
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

traindata_loader=DataLoader(
    dataset=train_data, batch_size=batch_size, shuffle=True,
    num_workers=num_workers, pin_memory=True, drop_last=True,
)



def train():
    logs_dir = 'logs/dada2k_seq96_b4_2kiter'
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    model_dir = os.path.join(logs_dir, "ckpts")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    logger = SummaryWriter(logs_dir, flush_secs=10)
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

    model = accident_frame(h_dim, n_layers, depth, adim, heads, num_tokens, c_dim, s_dim1, s_dim2, keral, num_class).to(
        device)

    opt1 = torch.optim.Adam([
        {'params': model.fusion.parameters(), 'lr': 1e-6},
        {'params': model.features.parameters(), 'lr': 1e-6},
        {'params': model.deconv.parameters(), 'lr': 1e-4},
        {'params': model.gru_net.parameters(), 'lr': 1e-5},
    ])

    scheduler1 = torch.optim.lr_scheduler.StepLR(opt1, step_size=1, gamma=0.1)

    # load ckpt to continue training
    ckpt_files = [f for f in os.listdir(model_dir) if f.endswith(".pth")]
    if ckpt_files not in ([], None):
        latest_ckpt = natsorted(os.listdir(model_dir))[-1]  # Get the latest checkpoint
        checkpoint_path = os.path.join(model_dir, latest_ckpt)
        print(f"Loading checkpoint {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)  # Ensure loading on correct device
        model.load_state_dict(checkpoint['model_state_dict'])
        opt1.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler1.load_state_dict(checkpoint['scheduler_state_dict'])
        # Move optimizer to the correct device
        for param_group in opt1.param_groups:
            param_group['lr'] = param_group['lr']
        start_epoch = checkpoint['epoch'] + 1  # Resume from next epoch
    else:
        print("No checkpoint found, training from scratch.")
        start_epoch = 0

    model.to(device)
    model.train()
    epoch = start_epoch
    epoch_loss = 0.0
    #scaler = torch.cuda.amp.GradScaler()
    for epoch_ in range(num_epochs):
        if epoch_ > 0:
            print("\n ====== Start new loop over the dataset ======\n")
        num_batches = len(traindata_loader)
        loop = tqdm(traindata_loader ,total = len(traindata_loader), leave = True)
        #sampler_train.set_epoch(epoch)
        for i, (imgs, focus, labels, info, texts) in enumerate(loop):
            loop.set_description(f"Epoch  [{epoch}/...]")

            assert not torch.isnan(imgs).any(), "NaN detected in input images!"
            assert not torch.isnan(focus).any(), "NaN detected in fixations!"
            assert not torch.isnan(labels).any(), "NaN detected in labels!"

            # print(imgs.shape)
            imgs = imgs.to(device)
            focus = focus.to(device)
            labels = np.array(labels).astype(int)
            labels = torch.from_numpy(labels)
            labels = labels.to(device)
        
            opt1.zero_grad()

            # with torch.cuda.amp.autocast():  # Enables mixed precision
            #     loss, outputs = model(imgs, focus, labels.long(), texts)
            #     loss = loss.mean()

            # scaler.scale(loss).backward()  # Scale the gradients to prevent underflow
            # # Unscale gradients before clipping
            # scaler.unscale_(opt1)  # Correct placement before clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 10)  
            # scaler.step(opt1)  # Update optimizer
            # scaler.update()  # Adjust scaling factor for stability

            loss, outputs = model(imgs, focus, labels.long(), texts)
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),10)
            opt1.step()

            loss = loss.item()
            loop.set_postfix(loss = loss)
            epoch_loss += loss
            logger.add_scalar('train/batch_loss', loss, epoch * nb_batches_per_epoch + i)
            if (i+1) % 50 == 0:
                gc.collect()  # Run garbage collection to clear unused memory
                torch.cuda.empty_cache()  # Free up CUDA memory
                print(f"Step {i}/{nb_batches_per_epoch}")
                utils.print_memory_usage()
            del loss
            del outputs
            if (i+1) % nb_batches_per_epoch == 0:
                logger.add_scalar('train/epoch_loss', epoch_loss / nb_batches_per_epoch, epoch)
                epoch_loss = 0.
                #
                if (epoch + 1) % 10 == 0:
                    scheduler1.step()
                # save model
                print("Saving model...")
                model_file = os.path.join(model_dir, f'ckpt_{epoch:02d}.pth')
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt1.state_dict(),
                    'scheduler_state_dict': scheduler1.state_dict()
                }
                torch.save(checkpoint, model_file)
                #
                epoch += 1
                if epoch == num_epochs:
                    break

    logger.close()
    print("Training is complete!")

if __name__ == "__main__":
    train()
