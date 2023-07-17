import os
from datetime import datetime
import argparse
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from apex.parallel import DistributedDataParallel as DDP
from apex import amp

from torch.utils.tensorboard import SummaryWriter

def main():
    print('run main')
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    # parser.add_argument('-nr', '--nr', default=0, type=int,
    #                     help='ranking within the nodes')
    parser.add_argument('--epochs', default=2, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--dist-backend',  default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--master-ip',  default='', type=str,
                        help='master node ip')
    parser.add_argument('--master-port',  default='', type=str,
                        help='master node port')
    parser.add_argument('--nccl-debug',  default='', type=str,
                        help='nccl log level: WARN, INFO, TRACE, VERSION')
    args = parser.parse_args()
    args.world_size = args.gpus * args.nodes

    if len(args.nccl_debug) > 0:
        os.environ['NCCL_DEBUG'] = args.nccl_debug

    if len(args.master_ip) > 0:
        os.environ['MASTER_ADDR'] = args.master_ip
    else:
        os.environ['MASTER_ADDR'] = os.environ['PAI_HOST_IP_worker_0']
    if len(args.master_port) > 0:
        os.environ['MASTER_PORT'] = args.master_port
    else:
        os.environ['MASTER_PORT'] = os.environ['PAI_worker_0_SynPort_PORT']

    print('master:', os.environ['MASTER_ADDR'], 'port:', os.environ['MASTER_PORT'])
    # Data loading code

    print ("Data download start...")
    trainset = torchvision.datasets.MNIST(root='./data',
                                          train=True,
                                          transform=transforms.ToTensor(),
                                          download=False)
    print ("Data download finish")
    mp.spawn(train, nprocs=args.gpus, args=(args, trainset))


class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

def train(gpu, args, trainset):
    pai_task_index = int(os.environ['PAI_TASK_INDEX'])
    print("start train task[%d] gpu[%d]" % (pai_task_index, gpu))
    tb_writer = SummaryWriter('/mnt/tensorboard')
    rank = pai_task_index * args.gpus + gpu
    dist.init_process_group(backend=args.dist_backend, init_method='env://', world_size=args.world_size, rank=rank)
    torch.manual_seed(0)
    model=ConvNet()
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    batch_size = 100
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)
    # Wrap the model
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    # Data loading code
    trainsampler = torch.utils.data.distributed.DistributedSampler(
        trainset,
        num_replicas=args.world_size,
        rank=rank,
        shuffle=True,
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=False, num_workers=2, sampler=trainsampler)

    # testset = torchvision.datasets.CIFAR10(
    #     root='./data', train=False, download=True, transform=transforms.ToTensor())
    # testloader = torch.utils.data.DataLoader(
    #     testset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, sampler=trainsampler)

    start = datetime.now()
    total_step = len(trainloader)
    for epoch in range(args.epochs):
        for i, (images, labels) in enumerate(trainloader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            if (i + 1) % 10 == 0:
                tb_writer.add_scalar("Train Loss", loss, epoch)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0 and gpu == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, args.epochs, i + 1, total_step,
                                                                         loss.item()))
    tb_writer.close()
    if gpu == 0:
        print("Training complete in: " + str(datetime.now() - start))


if __name__ == '__main__':
    main()