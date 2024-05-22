import torch
from tqdm.autonotebook import tqdm
from torchmetrics.functional import accuracy
import numpy as np
import pathlib
import os

class Blackbox:
    def __init__(self, net, device, train_loader, batch_size, max_epoch, learning_rate, criterion, optimizer,
                 val_loader, verbose = True):
        self.net = net.to(device)
        self.device = device
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.train_loader = train_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.val_loader = val_loader
        self.n_minibatches = len(self.train_loader)

        self.acc = False   # Compute accuracy or not

        self.ID = np.random.randint(0, high=2**31)

        if self.verbose:
            print("===== HYPERPARAMETERS =====")
            print("batch_size=", self.batch_size)
            print("n_epochs=", self.max_epoch)
            print("learning_rate=", self.learning_rate)
            print("=" * 30)

    def train_single_epoch(self):
        """
        Train the model for a single epoch
        :return
        """
        single_train_loss = 0

        # Compute the accuracy for multi-class classification
        if self.acc:
            single_train_acc = 0

        for i, (inputs, targets) in enumerate(self.train_loader):
            # Move tensors to correct device
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            # zero the parameter gradients
            self.optimizer.zero_grad()

            # forward + backward + optimize
            outputs = self.net(inputs)
            # targets = targets.to(torch.long) # needed for the multiclass classification cases
            loss = self.criterion(outputs, inputs, targets)
            loss.backward()
            self.optimizer.step()

            single_train_loss += loss.item()
            if self.acc:
                single_train_acc += float(accuracy(outputs,targets).cpu().numpy())

        single_train_loss /= len(self.train_loader)
        if self.acc:
          single_train_acc /= len(self.train_loader)

        return (single_train_loss, single_train_acc) if self.acc else single_train_loss

    def full_train(self, save_dir = './models'):
        pathlib.Path(os.path.dirname(save_dir)).mkdir(parents=True, exist_ok=True)
        # Initialize monitoring variables
        if self.acc:
          stats = {'epoch': [], 'train_loss':[], 'val_loss':[], 'train_acc':[],'val_acc':[]}
        else:
          stats = {'epoch': [], 'train_loss':[], 'val_loss':[]}

        print("Begin training.")
        for e in tqdm(range(1, self.max_epoch+1)):

            epoch_train_loss = 0
            epoch_val_loss = 0

            if self.acc:
              epoch_train_acc = 0
              epoch_val_acc = 0

            self.net.train()
            if self.acc:
              epoch_train_loss, epoch_train_acc = self.train_single_epoch()
            else:
              epoch_train_loss = self.train_single_epoch()

            # scheduler.step()
            self.net.eval()
            for inputs, targets in self.val_loader:
              inputs, targets = inputs.to(self.device), targets.to(self.device)
              outputs = self.net(inputs)
              val_loss = self.criterion(outputs, inputs, targets)
              epoch_val_loss += val_loss.item()
              if self.acc:
                val_acc = self.get_acc(inputs, targets)
                epoch_val_acc += val_acc

            epoch_val_loss /= len(self.val_loader)
            if self.acc:
              epoch_val_acc /= len(self.val_loader)


            stats['epoch'].append(e)
            stats['train_loss'].append(epoch_train_loss)
            stats['val_loss'].append(epoch_val_loss)
            if self.acc:
              stats['train_acc'].append(epoch_train_acc)
              stats['val_acc'].append(epoch_val_acc)


            if self.verbose:
              print(f'Epoch {e+0:03}: | train_loss: {epoch_train_loss:.3f} | ', end = '')
              print(f'val_loss: {epoch_val_loss:.3f} | ', end = '')
              print('', flush = True)
              if self.acc:
                print(f'Epoch {e+0:03}: | train_acc: {epoch_train_acc:.3f} | val_acc: {epoch_train_acc:.3f} | ', end='')


        saved_final_state = dict(stats=stats, model_state=self.net.state_dict())
        torch.save(saved_final_state, save_dir)
        return stats

    def get_acc(self, inputs, targets):
        self.net.eval()
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        with torch.no_grad():
            outputs = self.net(inputs)
            acc = float(accuracy(outputs,targets,top_k=1).cpu().numpy())

        return acc
    
