from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch
import numpy as np
import time
import os
import datetime
import argparse

from kitti_dataset import KittiDataset
import torch.nn.functional as F


from model import PointNet

SEED = 13
batch_size = 16
epochs = 100
decay_lr_factor = 0.95
decay_lr_every = 2
lr = 0.01
gpus = [0]
global_step = 0
show_every = 1
val_every = 3
date = datetime.date.today()
save_dir = "../output"


def save_ckp(ckp_dir, model, optimizer, epoch, best_acc, date):
  os.makedirs(ckp_dir, exist_ok=True)
  state = {
    'state_dict': model.state_dict(),
    'optimizer': optimizer.state_dict()
    }
  ckp_path = os.path.join(ckp_dir, f'date_{date}-epoch_{epoch}-maxacc_{best_acc:.3f}.pth')
  torch.save(state, ckp_path)
  torch.save(state, os.path.join(ckp_dir,f'latest.pth'))
  print('model saved to %s' % ckp_path)


def load_ckp(ckp_path, model, optimizer):
  state = torch.load(ckp_path)
  model.load_state_dict(state['state_dict'])
  optimizer.load_state_dict(state['optimizer'])
  print("model load from %s" % ckp_path)


def softXEnt(prediction, real_class):
    # TODO: return loss here
    
    target_n = torch.argmax(real_class, axis=1)
    
    #print("target shape: {}".format(target_n.size()))
    #print("shape of prediction: {}".format(prediction.size()))
    loss = F.nll_loss(prediction, target_n)
    #print("shape of loss: {}".format(loss.size()))
    #print("content of loss: {}".format(loss))
    #print("first pred: {}".format(prediction[0]))
    #print(np.argmax(loss.cpu().detach.numpy(),axis=1))
    #raise RuntimeError("xxx")
    return loss


def get_eval_acc_results(model, data_loader, device):
    """
    ACC
    """
    seq_id = 0
    model.eval()

    distribution = np.zeros([5])
    confusion_matrix = np.zeros([5, 5])
    pred_ys = []
    gt_ys = []
    with torch.no_grad():
        accs = []
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)

            # TODO: put x into network and get out
            out = model.forward(x)

            # TODO: get pred_y from out
            pred_y = np.argmax(out.cpu().detach().numpy(), axis=1)
            gt = np.argmax(y.cpu().numpy(), axis=1)

            # TODO: calculate acc from pred_y and gt
            acc = np.sum(pred_y == gt) / len(y)
            gt_ys = np.append(gt_ys, gt)
            pred_ys = np.append(pred_ys, pred_y)
            idx = gt

            accs.append(acc)

        return np.mean(accs)


def parse_args():
    parser = argparse.ArgumentParser(description="Arg parser")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--ckpt_save_interval", type=int, default=5)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument("--mode", type=str, default='train')
    parser.add_argument("--ckpt", type=str, default='None')

    parser.add_argument("--net", type=str, default='pointnet2_msg')

    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--lr_decay', type=float, default=0.2)
    parser.add_argument('--lr_clip', type=float, default=0.000001)
    parser.add_argument('--decay_step_list', type=list, default=[50, 70, 80, 90])
    parser.add_argument('--weight_decay', type=float, default=0.001)

    parser.add_argument("--output_dir", type=str, default='output')
    parser.add_argument("--extra_tag", type=str, default='default')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    #da_root = "./dataset/modelnet40_normal_resampled"
    args = parse_args()
    print("loading val dataset.")
    eval_set = KittiDataset( mode='EVAL', split='val')
    eval_loader = DataLoader(eval_set, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                             num_workers=args.workers, collate_fn=eval_set.collate_batch)


    writer = SummaryWriter('./output/runs/tersorboard')
    torch.manual_seed(SEED)
    device = torch.device(f'cuda:{gpus[0]}' if torch.cuda.is_available() else 'cpu')
    print("Loading train dataset...")
    
    train_set = KittiDataset( mode='TRAIN', split='train')
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                                  num_workers=args.workers, collate_fn=train_set.collate_batch)

    
    
    print("Set model and optimizer...")
    model = PointNet().to(device=device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(
          optimizer, step_size=decay_lr_every, gamma=decay_lr_factor)

    best_acc = 0.0
    model.train()
    print("Start trainning...")
    for epoch in range(epochs):
      acc_loss = 0.0
      num_samples = 0
      start_tic = time.time()
      for item in train_loader:
        print(item.keys())
        x = x.to(device)
        y = y.to(device)
        #print("trail feature shape: {}".format(x.size()))

        # TODO: set grad to zero
        optimizer.zero_grad()
        # TODO: put x into network and get out
        out = model(x)
        #print("is on cuda: {}".format(out.is_cuda))
        loss = softXEnt(out, y)
        
        # TODO: loss backward
        loss.backward()

        # TODO: update network's param
        optimizer.step()
        
        acc_loss += batch_size * loss.item()
        num_samples += y.shape[0]
        global_step += 1
        acc = np.sum(np.argmax(out.cpu().detach().numpy(), axis=1) == np.argmax(y.cpu().detach().numpy(), axis=1)) / len(y)
        # print('acc: ', acc)
        if (global_step + 1) % show_every == 0:
          # ...log the running loss
          writer.add_scalar('training loss', acc_loss / num_samples, global_step)
          writer.add_scalar('training acc', acc, global_step)
          # print( f"loss at epoch {epoch} step {global_step}:{loss.item():3f}, lr:{optimizer.state_dict()['param_groups'][0]['lr']: .6f}, time:{time.time() - start_tic: 4f}sec")
      scheduler.step()
      print(f"loss at epoch {epoch}:{acc_loss / num_samples:.3f}, lr:{optimizer.state_dict()['param_groups'][0]['lr']: .6f}, time:{time.time() - start_tic: 4f}sec")
      
      if (epoch + 1) % val_every == 0:
        
        acc = get_eval_acc_results(model, val_loader, device)
        print("eval at epoch[" + str(epoch) + f"] acc[{acc:3f}]")
        writer.add_scalar('validing acc', acc, global_step)

        if acc > best_acc:
          best_acc = acc
          save_ckp(save_dir, model, optimizer, epoch, best_acc, date)

          example = torch.randn(1, 3, 10000).to(device)
          traced_script_module = torch.jit.trace(model, example)
          traced_script_module.save("../output/traced_model.pt")