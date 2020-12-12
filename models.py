# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import warnings
warnings.simplefilter('ignore')
import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.init as init
import pytorch_lightning as pl
import pandas as pd

from torch.utils.data import DataLoader
from utils import *
from datasets import DoubleMotifs as PdDataset
from Blocks import *
from config import *

from ModernHopfield import Hopfield, StatePattern
import torchvision
import torchvision.transforms as transforms

from collections import OrderedDict
import wandb

def TransmuteBaseModel(ModelTemplate, TrainerTemplate):
    return ModelTemplate(TrainerTemplate)

class BinaryBindNetTemplate(pl.LightningModule):
    def __init__(self,):
        super(BinaryBindNetTemplate, self).__init__()
        self.hparams = None

    def forward(self, x):
        pass

    def prepare_data(self):

        OneHotDNA = lambda x: OneHotEncode(x, NucEnc)
        OneHotProtein = lambda x: OneHotEncode(x, AmEnc)

        MR = Memory_container(['rolls', 'prescribed'])

        preprocess_x = Compose([Pad(max_len=self.hparams['FIXED_LEN'], pad_symbol='Z'), OneHotProtein])
        preprocess_y = Compose([Upper, Pad(max_len=self.hparams['FIXED_LEN'], pad_symbol='X'), OneHotDNA])

        transform_x_train = Compose([to_tensor,
                                    Roll(max_roll=self.hparams['MAX_ROLL'], p=self.hparams['ROLL_prob'], memory_container=MR),
                                    #Prescribe(p=self.hparams['PREFIX_prob'], memory_container=MR),
                                    scatter_torch(len(AmEnc)+1),
                                    crop_out_padding, to_float])

        transform_x2_train = Compose([to_tensor,
                                    Roll(memory_container=MR, from_memory=True),
                                    #Prescribe(p=self.hparams['PREFIX_prob'], memory_container=MR),
                                    scatter_torch(len(NucEnc)+1),
                                    crop_out_padding, to_float])

        transform_y_train = Compose([to_tensor])

        # transform_y2_train = Compose([to_tensor,
        #                              Roll(memory_container=MR, from_memory=True),
        #                              PrescribeShadow(0, MR)])

        transform_x_test = Compose([to_tensor, scatter_torch(len(AmEnc)+1), crop_out_padding, to_float])
        transform_x2_test = Compose([to_tensor, scatter_torch(len(NucEnc)+1), crop_out_padding, to_float])
        transform_y_test = Compose([to_tensor])

        df_train = pd.read_csv(self.hparams['train_data'])  #('./data/seq_struct_train.csv')
        df_val = pd.read_csv(self.hparams['val_data'])
        df_test = pd.read_csv(self.hparams['test_data'])

        self.batch_augments = MR

        self.train_dataset = PdDataset(df_train, self.hparams['Positive_rate'], preprocess_x=preprocess_x, preprocess_y=preprocess_y,
                                   transforms=transform_x_train, transforms_y=transform_y_train, transforms_x2=transform_x2_train)
        self.val_dataset = PdDataset(df_val, self.hparams['Positive_rate'], preprocess_x=preprocess_x, preprocess_y=preprocess_y,
                                   transforms=transform_x_train, transforms_y=transform_y_train, transforms_x2=transform_x2_train)
        self.test_dataset = PdDataset(df_test, positive_rate=1, preprocess_x=preprocess_x, preprocess_y=preprocess_y,
                                   transforms=transform_x_test, transforms_y=transform_y_test, transforms_x2=transform_x2_test)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams['batch_size'], shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams['batch_size'], shuffle=False, num_workers=2)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams['batch_size'], shuffle=False, num_workers=2)

    def loss_function(self, y, target):
        BCE = F.binary_cross_entropy(input=y, target=target, reduction='mean')
        return BCE

    def training_step(self, batch, batch_idx):
        x1, x2, y, s, idx = batch
        out = self(x1, x2)
        loss = self.loss_function(out, y)

        labels_hat1 = torch.argmax(out, dim=1)
        acc1 = torch.sum(y == labels_hat1).item() / (len(y) * 1.0)


        log = {'loss': loss, 'train_acc1': acc1}

        output = OrderedDict({'loss': loss,
                              'train_acc': acc1,
                              'log': log
                              })
        return output

    def validation_step(self, batch, batch_idx):
        x1, x2, y, s, idx = batch
        out = self(x1, x2)
        loss = self.loss_function(out, y)

        labels_hat1 = torch.argmax(out, dim=1)
        acc1 = torch.sum(y == labels_hat1).item() / (len(y) * 1.0)

        log = {'val_loss': loss, 'val_acc': acc1}

        output = OrderedDict({'val_loss': loss,
                              'val_acc': acc1,
                              'log': log
                              })
        return output

    def test_step(self, batch, batch_idx):
        x, y1, y2, s, idx = batch
        out1, out2 = self(x)
        loss1 = self.loss_function(out1, y1)
        loss2 = self.loss_function(out2, y2)
        loss = loss1+loss2

        labels_hat1 = torch.argmax(out1, dim=1)
        acc1 = torch.sum(y1 == labels_hat1).item() / (len(y1) * 1.0)

        labels_hat2 = torch.argmax(out2, dim=1)
        acc2 = torch.sum(y2 == labels_hat2).item() / (len(y2) * 1.0)

        log = {'test_loss1': loss1, 'test_loss2': loss2, 'test_acc1': acc1, 'test_acc2': acc2}
        output = OrderedDict({'test_loss': loss,
                              'test_loss1': loss1,
                              'test_loss2': loss1,
                              'y_hat1': out1,
                              'y_hat2': out2,
                              'test_acc1': torch.tensor(acc1),
                              'test_acc2': torch.tensor(acc2), # everything must be a tensor
                              #'missclass': (x[miss_idx], labels_hat[miss_idx], y[miss_idx]),
                              'log': log
                              })
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams['optimizer']['learning_rate'],
                                     weight_decay=self.hparams['optimizer']['weight_decay'])
        return optimizer

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean().item()

        mean_acc1 = 0
        for output in outputs:
            mean_acc1 += output['train_acc']
        mean_acc1 /= len(outputs)


        tqdm_dict = {'avg_train_loss': avg_loss, 'train_acc1': mean_acc1}
        log = {'avg_train_loss': avg_loss, 'train_accuracy1': mean_acc1}
        results = {
            'train_loss': avg_loss,
            'progress_bar': tqdm_dict,
            'log': log
        }
        return results

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean().item()

        mean_acc1 = 0
        for output in outputs:
            mean_acc1 += output['val_acc']
        mean_acc1 /= len(outputs)

        tqdm_dict = {'avg_val_loss': avg_loss, 'val_acc': mean_acc1}
        log = {'avg_val_loss': avg_loss, 'val_accuracy': mean_acc1}
        results = {
            'val_loss': avg_loss,
            'progress_bar': tqdm_dict,
            'log': log
        }
        return results

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean().item()
        y_hats1 = torch.cat([x['y_hat1'] for x in outputs])
        y_hats2 = torch.cat([x['y_hat2'] for x in outputs])
        #ys = torch.cat([x['y'] for x in outputs])

        mean_acc1 = 0
        for output in outputs:
            mean_acc1 += output['test_acc1']
        mean_acc1 /= len(outputs)

        mean_acc2 = 0
        for output in outputs:
            mean_acc2 += output['test_acc2']
        mean_acc2 /= len(outputs)

        log = {'avg_test_loss': avg_loss, 'test_accuracy1': mean_acc1, 'test_accuracy2': mean_acc2}
        results = {
            'test_predictions1': y_hats1,
            'test_predictions2': y_hats2,
            #'test_targets': ys,
            'avg_test_loss': avg_loss,
            'log': log
        }
        return results

class BINDNETTemplate2(BinaryBindNetTemplate):
    def __init__(self,):
        super(BINDNETTemplate2, self).__init__()
        self.hparams = None

    def forward(self, x):
        pass

    def prepare_data(self):

        OneHotDNA = lambda x: OneHotPandasString(x, NucEnc)
        OneHotProtein = lambda x: OneHotPandasString(x, AmEnc)

        MR = Memory_container(['rolls', 'prescribed'])

        preprocess_x = Compose([Pad(max_len=self.hparams['FIXED_LEN'], pad_symbol='Z'), OneHotProtein])
        preprocess_y = Compose([Upper, Pad(max_len=self.hparams['FIXED_LEN'], pad_symbol='N'), OneHotDNA])

        transform_x_train = Compose([to_tensor,
                                     Roll(max_roll=self.hparams['MAX_ROLL'], p=self.hparams['ROLL_prob'],
                                          memory_container=MR),
                                     Prescribe(p=self.hparams['PREFIX_prob'], memory_container=MR),
                                     scatter_torch(len(AmEnc) + 1),
                                     crop_out_padding, to_float])

        transform_x2_train = Compose([to_tensor,
                                      Roll(max_roll=self.hparams['MAX_ROLL'], p=self.hparams['ROLL_prob'],
                                           memory_container=MR),
                                      Prescribe(p=self.hparams['PREFIX_prob'], memory_container=MR),
                                      scatter_torch(len(NucEnc) + 1),
                                      crop_out_padding, to_float])

        transform_y_train = Compose([to_tensor])

        # transform_y2_train = Compose([to_tensor,
        #                              Roll(memory_container=MR, from_memory=True),
        #                              PrescribeShadow(0, MR)])

        transform_x_test = Compose([to_tensor, scatter_torch(len(AmEnc) + 1), crop_out_padding, to_float])
        transform_x2_test = Compose([to_tensor, scatter_torch(len(NucEnc) + 1), crop_out_padding, to_float])
        transform_y_test = Compose([to_tensor])

        df_train = pd.read_json(self.hparams['train_data'], lines=True)  # ('./data/seq_struct_train.csv')
        df_val = pd.read_json(self.hparams['val_data'], lines=True)
        df_test = pd.read_json(self.hparams['test_data'], lines=True)

        self.batch_augments = MR

        self.train_dataset = PdDataset(df_train, preprocess_x=preprocess_x, preprocess_y=preprocess_y,
                                       transforms=transform_x_train, transforms_y=transform_y_train,
                                       transforms_x2=transform_x2_train)
        self.val_dataset = PdDataset(df_val, preprocess_x=preprocess_x, preprocess_y=preprocess_y,
                                     transforms=transform_x_train, transforms_y=transform_y_train,
                                     transforms_x2=transform_x2_train)
        self.test_dataset = PdDataset(df_test, preprocess_x=preprocess_x, preprocess_y=preprocess_y,
                                      transforms=transform_x_test, transforms_y=transform_y_test,
                                      transforms_x2=transform_x2_test)
        self.set_loss_functions()

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams['batch_size'], shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams['batch_size'], shuffle=False, num_workers=2)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams['batch_size'], shuffle=False, num_workers=2)

    def set_loss_functions(self):
        CE = torch.nn.CrossEntropyLoss(reduction='mean')
        self.Losses = [CE, CE, CE]

    @classmethod
    def MCRMSE(cls, prds, tgts, weight=None):
        mse_loss = torch.nn.MSELoss()
        if weight is None:
            return (torch.sqrt(mse_loss(prds[:, 0, :68], tgts[:, 0, :68])) +
                    torch.sqrt(mse_loss(prds[:, 1, :68], tgts[:, 1, :68])) +
                    torch.sqrt(mse_loss(prds[:, 2, :68], tgts[:, 2, :68]))) / 3


    def training_step(self, batch, batch_idx):
        x, *Y, L, bpp = batch
        *O, emb = self(x, bpp)
        losses = []
        accs = []
        for o,y,L in zip(O, Y, self.Losses):
            losses.append(L(o, y))
        loss = sum(losses)

        for i in range(len(Y)-1):
            predict = torch.argmax(O[i], dim=1)
            acc = torch.sum(Y[i] == predict).item() / (len(Y[i]) * 1.0)
            accs.append(acc)


        log = {'loss': loss}
        output = OrderedDict({'loss': loss, 'log': log})
        for i in range(len(losses)):
            log[f'train_loss{i+1}'] = losses[i]
            output[f'train_loss{i+1}'] = losses[i]

        for i in range(len(accs)):
            log[f'train_acc{i+1}'] = accs[i]
            output[f'train_acc{i+1}'] = accs[i]

        return output

    def validation_step(self, batch, batch_idx):
        x, *Y, L, bpp = batch
        *O, emb = self(x, bpp)
        losses = []
        accs = []
        for o,y,L in zip(O, Y, self.Losses):
            losses.append(L(o, y))
        loss = sum(losses)

        for i in range(len(Y)-1):
            predict = torch.argmax(O[i], dim=1)
            acc = torch.sum(Y[i] == predict).item() / (len(Y[i]) * 1.0)
            accs.append(acc)

        log = {'loss': loss}
        output = OrderedDict({'loss': loss, 'log': log})
        for i in range(len(losses)):
            log[f'val_loss{i + 1}'] = losses[i]
            output[f'val_loss{i + 1}'] = losses[i]

        for i in range(len(accs)):
            log[f'val_acc{i + 1}'] = accs[i]
            output[f'val_acc{i + 1}'] = accs[i]

        return output

    def test_step(self, batch, batch_idx):
        x, *Y, L, bpp = batch
        *O, emb = self(x, bpp)
        losses = []
        accs = []
        for o, y, L in zip(O, Y, self.Losses):
            losses.append(L(o, y))
        loss = sum(losses)

        for i in range(len(Y)-1):
            predict = torch.argmax(O[i], dim=1)
            acc = torch.sum(Y[i] == predict).item() / (len(Y[i]) * 1.0)
            accs.append(acc)

        log = {'loss': loss}
        output = OrderedDict({'loss': loss, 'log': log})
        for i in range(len(losses)):
            log[f'test_loss{i + 1}'] = losses[i]
            log[f'test_acc{i + 1}'] = accs[i]
            output[f'test_loss{i + 1}'] = losses[i]
            output[f'test_acc{i + 1}'] = accs[i]
            output[f'test_output{i+1}'] = O[i]
        output['test_embedding'] = emb
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams['learning_rate'],
                                     weight_decay=self.hparams['weight_decay'])
        # lr_scheduler = {
        #     'scheduler': torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.hparams['max_learning_rate'],
        #                                                      **self.hparams['scheduler']),
        #     'name': 'current_learning_rate'}
        #return optimizer [lr_scheduler]

        return optimizer

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean().item()
        avg_losses = [torch.stack([x[f'train_loss{i+1}'] for x in outputs]).mean().item() for i in range(len(self.Losses))]
        mean_accs = []

        tqdm_dict = {'avg_train_loss': avg_loss}
        log = {'avg_train_loss': avg_loss}
        for i in range(len(self.Losses)):
            log[f'avg_train_loss{i + 1}'] = avg_losses[i]

        for i in range(len(self.Losses)-1):
            mean_acc = 0
            for output in outputs:
                mean_acc += output[f'train_acc{i+1}']
            mean_acc /= len(outputs)
            mean_accs.append(mean_acc)
            log[f'avg_train_accuracy{i+1}'] = mean_acc

        results = {
            'train_loss': avg_loss,
            'progress_bar': tqdm_dict,
            'log': log
        }
        return results

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean().item()
        avg_losses = [torch.stack([x[f'val_loss{i+1}'] for x in outputs]).mean().item() for i in range(len(self.Losses))]
        mean_accs = []

        tqdm_dict = {'avg_val_loss': avg_loss}
        log = {'avg_val_loss': avg_loss}
        for i in range(len(self.Losses)):
            log[f'avg_val_loss{i+1}'] = avg_losses[i]

        for i in range(len(self.Losses)-1):
            mean_acc = 0
            for output in outputs:
                mean_acc += output[f'val_acc{i+1}']
            mean_acc /= len(outputs)
            mean_accs.append(mean_acc)
            log[f'avg_val_accuracy{i+1}'] = mean_acc

        results = {
            'val_loss': avg_loss,
            'progress_bar': tqdm_dict,
            'log': log
        }
        return results

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean().item()
        avg_losses = [torch.stack([x[f'test_loss{i+1}'] for x in outputs]).mean().item() for i in range(len(self.Losses))]
        mean_accs = []

        tqdm_dict = {'avg_test_loss': avg_loss}
        log = {'avg_test_loss': avg_loss}
        for i in range(len(self.Losses)-1):
            mean_acc = 0
            for output in outputs:
                mean_acc += output[f'test_acc{i+1}']
            mean_acc /= len(outputs)
            mean_accs.append(mean_acc)
            log[f'avg_test_accuracy{i+1}'] = mean_acc
            log[f'avg_test_loss{i+1}'] = avg_losses[i]

        results = {
            'test_loss': avg_loss,
            'progress_bar': tqdm_dict,
            'log': log
        }
        return results

class COUPLER(BinaryBindNetTemplate):
    def __init__(self, hparams):
        super(COUPLER, self).__init__()
        self.hparams = hparams
        FLen = hparams['FIXED_LEN']
        input_dim1 = self.hparams['input_dim1']
        input_dim2 = self.hparams['input_dim2']
        main_hid1 = self.hparams['main_hidden1']
        main_hid2 = self.hparams['main_hidden2']
        second_hid = self.hparams['second_hidden']
        gr = self.hparams['groups']

        self.ProteinEncoder = SeqEncoder(input_dim1, main_hid1, gr, second_hid=256)
        self.DNAEncoder = SeqEncoder(input_dim2, main_hid2, gr, second_hid=256)

        self.Cross = HopfieldConvDecoder(main_hid1, main_hid2, second_hid, FLen//2, maxpool_kernel=1,
                 stride=1, groups=2, beta=1.0, activation='selu', norm='BN', dropout=0.0,
                 weight_standard=True)

        self.OutPool = nn.AvgPool1d(FLen//2)
        self.OutLayer = nn.Linear(2*second_hid+main_hid1+main_hid2, 1)

        self.apply(weights_init)


    def forward(self, x1, x2):
        x1 = self.ProteinEncoder(x1).permute(1,2,0)
        x2 = self.DNAEncoder(x2).permute(1,2,0)

        x12, x22 = self.Cross(x1, x2)

        x = F.dropout(F.relu(self.OutPool(torch.cat((x1, x2, x12, x22), dim=1)))).squeeze()

        y = self.OutLayer(x)

        return F.sigmoid(y)

class SYGIL(BinaryBindNetTemplate):
    def __init__(self, hparams):
        super(SYGIL, self).__init__()
        self.hparams = hparams
        FLen = hparams['FIXED_LEN']
        main_hid = self.hparams['main_hidden']
        second_hid = self.hparams['second_hidden']
        channel_pooled = self.hparams['channel_pooled']
        gr = self.hparams['groups']

        self.ConvTrunk = ConvTrunk(in_channels=4, hid_channels=main_hid, groups=gr, activation='relu')
        self.AxialNet = AxialNet(in_channels=main_hid, hid_channels=second_hid,
                                 length=FLen // 2, kernel_sizes=(20, 10),
                                 groups=gr, activation='relu')

        self.HopfieldGlobal = HopfieldConvNet(in_channels=main_hid, hid_channels=channel_pooled,
                                              length=FLen // 2,
                                              hidden_lengths=[FLen // 4, FLen // 10, FLen // 20],
                                              kernel_sizes=(2, 5, 10), strides=(2, 5, 10),
                                              num_memories=(200, 100, 50),
                                              groups=gr, beta=1.25, norm=True,
                                              activation='relu')

        self.BoringNet = HighColumnNet(d_model=main_hid, hid_channels=channel_pooled,
                                       num_memories=100, groups=gr, activation='relu', dim_feedforward=256)

        self.MainConv = nn.Conv1d(4 * main_hid, self.hparams['terminal_hidden'], kernel_size=1, stride=1, padding=0)
        self.BNorm1 = nn.BatchNorm1d(self.hparams['terminal_hidden'])

        self.UpConv1 = nn.ConvTranspose1d(self.hparams['terminal_hidden'], self.hparams['terminal_hidden'],
                                          kernel_size=2, stride=2, padding=0)
        self.GroupNorm2 = nn.GroupNorm(num_groups=gr, num_channels=self.hparams['terminal_hidden'])

        self.conv5 = nn.Conv1d(self.hparams['terminal_hidden'] + main_hid, main_hid, kernel_size=1, dilation=1,
                               padding=0)
        self.BNorm3 = nn.BatchNorm1d(main_hid)
        self.OutLayer = nn.Conv1d(main_hid, hparams['output_dim_total'], kernel_size=1, padding=0)
        self.RZ = nn.ParameterList([nn.Parameter(torch.Tensor([0.1])) for _ in range(7)])

        self.apply(weights_init)

    def load_postprocessnet(self, PN=False):
        if not PN:
            print('Loading Standard postprocess net')
            self.PostprocessNet = PostProcessNetSimple(self.hparams['main_hidden'], self.hparams['main_hidden'], out_dim=5)
        else:
            print('Attaching provided PN')
            self.PostprocessNet = PN

    def forward(self, x, bpp):
        x0, x00 = self.ConvTrunk(x)

        x1 = self.AxialNet(x0)
        x2 = self.HopfieldGlobal(x0)
        x3 = self.BoringNet(x0)

        x = torch.cat([x0, x1, x2, x3], dim=1)

        x = F.dropout(self.BNorm1(F.leaky_relu(self.MainConv(x))), p=0.3)
        emb = F.dropout(self.GroupNorm2(F.leaky_relu(self.UpConv1(x))), p=0.2)

        x = torch.cat([emb, x00], dim=1)
        x = F.dropout(self.BNorm3(F.leaky_relu(self.conv5(x))), p=0.1)

        y = self.OutLayer(x)
        y1 = y[:, :self.hparams['output_dim1'], :]
        y2 = y[:, self.hparams['output_dim1']:self.hparams['output_dim2'], :]
        y3 = y[:, self.hparams['output_dim2']:self.hparams['output_dim3'], :]

        y4 = self.PostprocessNet(emb, bpp, y)
        return y1, y2, y3, y4, emb

class PostProcessNetSimple(nn.Module):
    def __init__(self, in_dim=96, main_hid=96, out_dim=5):
        super(PostProcessNetSimple, self).__init__()
        self.ffl0 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=(3, 3), padding=1)

        t = torch.Tensor([[[[3, 7, 3], [12, 40, 12], [3, 7, 3]]]])
        self.ffl0.weight = nn.Parameter(t.repeat(2, 1, 1, 1))

        self.SimpleConv1 = nn.Conv1d(in_dim, main_hid, kernel_size=1)
        self.SelfAttn1 = nn.TransformerEncoderLayer(d_model=main_hid, nhead=4, dim_feedforward=512, dropout=0.1)
        self.SelfAttn2 = nn.TransformerEncoderLayer(d_model=main_hid, nhead=4, dim_feedforward=512, dropout=0.1)
        # self.SelfAttn1 = RZTXEncoderLayer(d_model=main_hid, nhead=4, dim_feedforward=512, dropout=0.1)
        # self.SelfAttn2 = RZTXEncoderLayer(d_model=main_hid, nhead=4, dim_feedforward=512, dropout=0.1)
        self.OutProjection = nn.Conv1d(main_hid, out_dim, kernel_size=1)


    def forward(self, emb, bpp, emb2=None):

        b, c, l = emb.shape
        bpp = torch.log(bpp + 1e-5) + 4
        bpp = bpp.type(torch.float32)
        bpp1 = bpp.unsqueeze(1).expand(b, 2, l, l)
        bpp2 = bpp.unsqueeze(1)
        bpp2 = self.ffl0(bpp2)
        bpp = torch.cat([bpp1, bpp2], dim=1).view(b * 4, l, l)


        x = self.SimpleConv1(emb)

        x = x.permute(2,0,1)
        #print(x.shape)
        x = self.SelfAttn1(x)
        #x = self.SelfAttn2(x, src_mask=(bpp/10))
        x = x.permute(1,2,0)

        y = self.OutProjection(x)
        return y

class PostProcessNetFinal(nn.Module):
    def __init__(self, in_dim=96, main_hid=96, out_dim=5):
        super(PostProcessNetFinal, self).__init__()
        self.ffl0 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), padding=1)

        t = torch.Tensor([[[[2, 7, 2], [10, 40, 10], [2, 7, 2]]]])
        self.ffl0.weight = nn.Parameter(t.repeat(1, 1, 1, 1))

        self.FirstProject = nn.Conv1d(in_dim, main_hid, kernel_size=1, groups=4)
        self.BN0 = nn.GroupNorm(num_groups=4, num_channels=main_hid)

        self.SelfAttn1 = Hopfield(main_hid, hidden_size=48, output_size=main_hid, batch_first=True,
                                  add_zero_association=True, num_heads=2, update_steps_max=1, dropout=0.1)
        self.SelfAttn2 = Hopfield(main_hid, hidden_size=48, output_size=main_hid, batch_first=True,
                                  add_zero_association=True, num_heads=2, update_steps_max=1, dropout=0.1)

        self.PatternSearch = HopfieldConvNet(in_channels=main_hid, hid_channels=48,
                                              length=200,
                                              hidden_lengths=[200 // 10, 200 // 5, 200 // 2],
                                              kernel_sizes=(10, 5, 2), strides=(10, 5, 2),
                                              num_memories=(200, 200, 100),
                                              groups=4, beta=1.2, norm=True,
                                              activation='leaky_relu', dropout=0.1)

        self.SimpleConv1 = nn.Conv1d(main_hid, main_hid, kernel_size=1)
        self.BN1 = nn.BatchNorm1d(main_hid)
        self.SimpleConv2 = nn.Conv1d(main_hid+18, main_hid, kernel_size=1)
        self.BN2 = nn.BatchNorm1d(main_hid)
        self.OutProjection = nn.Conv1d(main_hid, out_dim, kernel_size=1)
        self.RZ = nn.ParameterList([nn.Parameter(torch.Tensor([0.8])) for _ in range(4)])


    def forward(self, emb, bpp, emb2=None):

        b, c, l = emb.shape
        bpp = torch.log(bpp + 1e-5) + 4.5
        bpp = bpp.type(torch.float32)
        bpp1 = bpp.unsqueeze(1)#.expand(b, 2, l, l)
        bpp2 = bpp.unsqueeze(1)
        bpp2 = self.ffl0(bpp2)
        bpp = torch.cat([bpp1, bpp2], dim=1).view(b * 2, l, l)

        emb = F.dropout(self.BN0(F.relu(self.FirstProject(emb))), 0.15)

        x = emb.transpose(2,1)
        x = x + self.RZ[0]*self.SelfAttn1(x, association_mask=bpp)
        x = x + self.RZ[1]*self.PatternSearch(x.transpose(2,1)).transpose(2,1)
        x = x + self.RZ[2]*self.SelfAttn2(x, association_mask=bpp/5)
        x = x.transpose(2,1)

        x = F.dropout(self.BN1(F.relu(self.SimpleConv1(x))), 0.15)
        xs = torch.cat([x, emb2], dim=1)
        x = x + self.RZ[3]*F.dropout(self.BN2(F.relu(self.SimpleConv2(xs))), 0.1)

        y = self.OutProjection(x)

        return y


def weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
        init.kaiming_normal_(m.weight)

def StatNet_deprecated(stat):
    mu = stat.mean(dim=-1, keepdim=True)
    sigm = stat.std(dim=-1, keepdim=True)
    skew = torch.pow((stat - mu) / sigm, exponent=3).mean(dim=-1)
    kurt = torch.pow((stat - mu) / sigm, exponent=4).mean(dim=-1)
    p1 = stat ** 1.2 / (stat ** 1.2).sum(dim=-1, keepdim=True)
    p2 = stat ** 2 / (stat ** 2).sum(dim=-1, keepdim=True)
    pexp = torch.exp(stat) / torch.exp(stat).sum(dim=-1, keepdim=True)

    S1 = torch.mean(p1 * torch.log(p1), dim=-1)
    S2 = torch.mean(p2 * torch.log(p2), dim=-1)
    S3 = torch.mean(pexp * torch.log(pexp), dim=-1)
    stat = torch.cat([mu.squeeze(), sigm.squeeze(), skew, kurt, S1, S2, S3], dim=1)
    stat = F.leaky_relu(stat)
    return stat