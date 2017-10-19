import torch
import torch.nn as nn

from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.optim as optim
import model
import sampler
import time

import argparse


from inceptionresnetv2 import inceptionresnetv2

LookupChoices = type('', (argparse.Action, ), dict(__call__ = lambda a, p, n, v, o: setattr(n, a.dest, a.choices[v])))
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', help='path to the training images')
parser.add_argument('--epochs', default = 150, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--model', choices = dict(liftedstruct = model.LiftedStruct, triplet = model.Triplet, tripletratio = model.TripletRatio, pddm = model.Pddm, untrained = model.Untrained, margin = model.Margin), default = model.Margin, action = LookupChoices)
parser.add_argument('--sampler', choices = dict(simple = sampler.simple, triplet = sampler.triplet, npairs = sampler.npairs), default = sampler.npairs, action = LookupChoices)
parser.add_argument('--threads', default = 16, type = int)
args = parser.parse_args()

trans = transforms.Compose([transforms.Scale((160, 160)), transforms.ToTensor()])
faceset = ImageFolder(root=args.data_dir, transform=trans)
adapt_sampler = lambda batch, dataset, sampler, **kwargs: type('', (), dict(__len__ = dataset.__len__, __iter__ = lambda _: itertools.chain.from_iterable(sampler(batch, dataset, **kwargs))))()

train_loader = torch.utils.data.DataLoader(faceset, sampler = adapt_sampler(opts.batch, dataset_train, opts.sampler), num_workers = opts.threads, batch_size = opts.batch, drop_last = True)

base_model = inceptionresnetv2()
model = args.model(base_model, len(faceset.classes)).cuda()
optimizer = optim.Adam(base_model.parameters(), **model.optimizer_params)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **model.lr_scheduler_params)

for epoch in range(args.epochs):
    scheduler.step()
    model.train()
    loss_all = []
    for batch_idx, batch in enumerate(train_loader):
        tic = time.time()
        images, labels = batch
        images, labels = Variable(images.cuda()), Variable(labels.cuda())
        loss = model.criterion(model(images), labels)
        loss_all.append(loss.data[0])
        optimizer.zero_grad()
        loss.backward()
        print('train {:>3}.{:05}  loss  {:.04f}   hz {:.02f}'.format(epoch, batch_idx, loss_all[-1], len(images) / (time.time() - tic)))
        optimizer.step()

