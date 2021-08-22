import os
import matplotlib.pyplot as plt
import datetime
import io
from PIL import Image
from torchvision.transforms import ToTensor
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from models.model_encdec import model_encdec
import dataset_invariance
from torch.autograd import Variable
import tqdm

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

class Trainer:
    def __init__(self, config):
        """
        The Trainer class handles the training procedure for training the autoencoder.
        :param config: configuration parameters (see train_ae.py)
        """

        # test folder creating
        self.name_test = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.folder_tensorboard = 'runs/runs-ae/'
        self.folder_test = 'training/training_ae/' + self.name_test + '_' + config.info
        if not os.path.exists(self.folder_test):
            os.makedirs(self.folder_test)
        self.folder_test = self.folder_test + '/'
        self.file = open(self.folder_test + "details.txt", "w")

        print('Creating dataset...')
        tracks = json.load(open(config.dataset_file))
        self.dim_clip = 180
        self.data_train = dataset_invariance.TrackDataset(tracks,
                                                          len_past=config.past_len,
                                                          len_future=config.future_len,
                                                          train=True,
                                                          dim_clip=self.dim_clip)
        self.train_loader = DataLoader(self.data_train,
                                       batch_size=config.batch_size,
                                       num_workers=1,
                                       shuffle=True
                                       )
        self.data_test = dataset_invariance.TrackDataset(tracks,
                                                         len_past=config.past_len,
                                                         len_future=config.future_len,
                                                         train=False,
                                                         dim_clip=self.dim_clip)
        self.test_loader = DataLoader(self.data_test,
                                      batch_size=config.batch_size,
                                      num_workers=1,
                                      shuffle=False
                                      )
        print('Dataset created')

        self.settings = {
            "batch_size": config.batch_size,
            "use_cuda": config.cuda,
            "dim_feature_tracklet": config.past_len * 2,
            "dim_feature_future": config.future_len * 2,
            "d_model": config.d_model,
            "past_len": config.past_len,
            "future_len": config.future_len,
            "num_prediction": config.num_prediction,
        }
        self.max_epochs = config.max_epochs

        # model
        self.mem_n2n = model_encdec(self.settings)

        # loss
        self.criterionLoss = nn.MSELoss()

        self.opt = torch.optim.Adam(self.mem_n2n.parameters(), lr=config.learning_rate)
        self.iterations = 0
        if config.cuda:
            self.criterionLoss = self.criterionLoss.cuda()
            self.mem_n2n = self.mem_n2n.cuda()
        self.start_epoch = 0
        self.config = config

        # Write details to file
        self.write_details()
        self.file.close()

        # Tensorboard summary: configuration
        self.writer = SummaryWriter(self.folder_tensorboard + self.name_test + '_' + config.info)
        self.writer.add_text('Training Configuration', 'model name: {}'.format(self.mem_n2n.name_model), 0)
        self.writer.add_text('Training Configuration', 'dataset train: {}'.format(len(self.data_train)), 0)
        self.writer.add_text('Training Configuration', 'dataset test: {}'.format(len(self.data_test)), 0)
        self.writer.add_text('Training Configuration', 'batch_size: {}'.format(self.config.batch_size), 0)
        self.writer.add_text('Training Configuration', 'learning rate init: {}'.format(self.config.learning_rate), 0)
        self.writer.add_text('Training Configuration', 'd_model: {}'.format(self.config.d_model), 0)

    def write_details(self):
        """
        Serialize configuration parameters to file.
        """

        self.file.write('points of past track: {}'.format(self.config.past_len) + '\n')
        self.file.write('points of future track: {}'.format(self.config.future_len) + '\n')
        self.file.write('train size: {}'.format(len(self.data_train)) + '\n')
        self.file.write('test size: {}'.format(len(self.data_test)) + '\n')
        self.file.write('batch size: {}'.format(self.config.batch_size) + '\n')
        self.file.write('learning rate: {}'.format(self.config.learning_rate) + '\n')
        self.file.write('embedding dim: {}'.format(self.config.d_model) + '\n')

    def draw_track(self, past, future, pred=None, index_tracklet=0, num_epoch=0, train=False):
        """
        Plot past and future trajectory and save it to tensorboard.
        :param past: the observed trajectory
        :param future: ground truth future trajectory
        :param pred: predicted future trajectory
        :param index_tracklet: index of the trajectory in the dataset (default 0)
        :param num_epoch: current epoch (default 0)
        :param train: True or False, indicates whether the sample is in the training or testing set
        :return: None
        """

        fig = plt.figure()
        past = past.cpu().numpy()
        future = future.cpu().numpy()
        plt.plot(past[:, 0], past[:, 1], c='blue', marker='o', markersize=3)
        plt.plot(future[:, 0], future[:, 1], c='green', marker='o', markersize=3)
        if pred is not None:
            pred = pred.cpu().numpy()
            plt.plot(pred[:, 0], pred[:, 1], color='red', linewidth=1, marker='o', markersize=1)
        plt.axis('equal')

        # Save figure in Tensorboard
        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg')
        buf.seek(0)
        image = Image.open(buf)
        image = ToTensor()(image).unsqueeze(0)

        if train:
            self.writer.add_image('Image_train/track' + str(index_tracklet), image.squeeze(0), num_epoch)
        else:
            self.writer.add_image('Image_test/track' + str(index_tracklet), image.squeeze(0), num_epoch)

        plt.close(fig)

    def fit(self):
        """
        Autoencoder training procedure. The function loops over the data in the training set max_epochs times.
        :return: None
        """
        config = self.config
        # Training loop
        for epoch in range(self.start_epoch, config.max_epochs):

            print(' ----- Epoch: {}'.format(epoch))
            loss = self._train_single_epoch()
            print('Loss: {}'.format(loss))

            if (epoch + 1) % 20 == 0:
                print('test on train dataset')
                # dict_metrics_train = self.evaluate(self.train_loader, epoch + 1)

                print('test on TEST dataset')
                dict_metrics_test = self.greedy_evaluate(self.test_loader, epoch + 1)

                # Tensorboard summary: learning rate
                for param_group in self.opt.param_groups:
                    self.writer.add_scalar('learning_rate', param_group["lr"], epoch)

                # Tensorboard summary: train
                # self.writer.add_scalar('accuracy_train/eucl_mean', dict_metrics_train['eucl_mean'], epoch)
                # self.writer.add_scalar('accuracy_train/Horizon10s', dict_metrics_train['horizon10s'], epoch)
                # self.writer.add_scalar('accuracy_train/Horizon20s', dict_metrics_train['horizon20s'], epoch)
                # self.writer.add_scalar('accuracy_train/Horizon30s', dict_metrics_train['horizon30s'], epoch)
                # self.writer.add_scalar('accuracy_train/Horizon40s', dict_metrics_train['horizon40s'], epoch)

                # Tensorboard summary: test
                self.writer.add_scalar('accuracy_test/eucl_mean', dict_metrics_test['eucl_mean'], epoch)
                self.writer.add_scalar('accuracy_test/Horizon10s', dict_metrics_test['horizon10s'], epoch)
                self.writer.add_scalar('accuracy_test/Horizon20s', dict_metrics_test['horizon20s'], epoch)
                self.writer.add_scalar('accuracy_test/Horizon30s', dict_metrics_test['horizon30s'], epoch)
                self.writer.add_scalar('accuracy_test/Horizon40s', dict_metrics_test['horizon40s'], epoch)

                # Save model checkpoint
                torch.save(self.mem_n2n, self.folder_test + 'model_ae_epoch_' + str(epoch) + '_' + self.name_test)
                self.save_results(dict_metrics_test, epoch=epoch + 1)

                # Tensorboard summary: model weights
                for name, param in self.mem_n2n.named_parameters():
                    self.writer.add_histogram(name, param.data, epoch)

        # Save final trained model
        torch.save(self.mem_n2n, self.folder_test + 'model_ae_' + self.name_test)

    def evaluate(self, loader, epoch=0):
        """
        Evaluate the model.
        :param loader: pytorch dataloader to loop over the data
        :param epoch: current epoch (default 0)
        :return: a dictionary with performance metrics
        """

        eucl_mean = horizon10s = horizon20s = horizon30s = horizon40s = 0
        dict_metrics = {}

        # Loop over samples
        for step, (index, past, future, presents, angle_presents, videos, vehicles, number_vec, scene, scene_one_hot) \
                in enumerate(tqdm.tqdm(loader)):
            past = Variable(past)
            future = Variable(future)
            if self.config.cuda:
                past = past.cuda()
                future = future.cuda()

            output = self.mem_n2n(past, future).data
            future_repeat = future.unsqueeze(1).repeat(1, self.config.num_prediction, 1, 1)
            distances = torch.norm(output - future_repeat, dim=3)
            distances_mean = torch.mean(distances, dim=2)
            index_min = torch.argmin(distances_mean, dim=1)
            best_pred = output[torch.arange(past.shape[0]), index_min[:]]

            distances = torch.norm(best_pred - future, dim=2)
            eucl_mean += torch.sum(torch.mean(distances, 1))
            horizon10s += torch.sum(distances[:, 9])
            horizon20s += torch.sum(distances[:, 19])
            horizon30s += torch.sum(distances[:, 29])
            horizon40s += torch.sum(distances[:, 39])

            # Draw sample: the first of the batch
            # if loader == self.test_loader:
            #     self.draw_track(past[0],
            #                     future[0],
            #                     preds[0],
            #                     index_tracklet=step,
            #                     num_epoch=epoch,
            #                     train=False
            #                     )

        dict_metrics['eucl_mean'] = eucl_mean / len(loader.dataset)
        dict_metrics['horizon10s'] = horizon10s / len(loader.dataset)
        dict_metrics['horizon20s'] = horizon20s / len(loader.dataset)
        dict_metrics['horizon30s'] = horizon30s / len(loader.dataset)
        dict_metrics['horizon40s'] = horizon40s / len(loader.dataset)

        return dict_metrics

    def greedy_evaluate(self, loader, epoch=0):
        """
        Evaluate the model.
        :param loader: pytorch dataloader to loop over the data
        :param epoch: current epoch (default 0)
        :return: a dictionary with performance metrics
        """

        eucl_mean = ADE_1s = ADE_2s = ADE_3s = horizon10s = horizon20s = horizon30s = horizon40s = 0

        dict_metrics = {}

        # Loop over samples
        for step, (index, past, future, presents, angle_presents, videos, vehicles, number_vec, scene, scene_one_hot) \
                in enumerate(tqdm.tqdm(loader)):
            dim_batch = past.size(0)
            past = Variable(past)
            future = Variable(future)
            if self.config.cuda:
                past = past.cuda()
                future = future.cuda()

            present_temp = past[:, -1, :2].unsqueeze(1)
            preds = present_temp.repeat_interleave(self.config.num_prediction, dim=2)
            past_embeded = self.mem_n2n.encode(past).data

            for i in range(self.config.future_len - 1):
                # tgt_mask = generate_square_subsequent_mask(preds.shape[1])
                # if self.config.cuda:
                #     tgt_mask = tgt_mask.cuda()
                # raise
                output = self.mem_n2n.decode(preds, past_embeded).data
                output = output.permute(1, 0, 2)
                output = self.mem_n2n.FC_output(output).data
                preds = torch.cat((preds, output[:, -1:, :]), 1)

            output = preds.view(dim_batch, self.config.num_prediction, self.config.future_len, 2)
            future_repeat = future.unsqueeze(1).repeat(1, self.config.num_prediction, 1, 1)
            distances = torch.norm(output - future_repeat, dim=3)
            distances_mean = torch.mean(distances, dim=2)
            index_min = torch.argmin(distances_mean, dim=1)
            best_pred = output[torch.arange(past.shape[0]), index_min[:]]
            distances = torch.norm(best_pred - future, dim=2)

            eucl_mean += torch.sum(torch.mean(distances, 1))
            ADE_1s += torch.sum(torch.mean(distances[:, :10], 1))
            ADE_2s += torch.sum(torch.mean(distances[:, :20], 1))
            ADE_3s += torch.sum(torch.mean(distances[:, :30], 1))
            horizon10s += torch.sum(distances[:, 9])
            horizon20s += torch.sum(distances[:, 19])
            horizon30s += torch.sum(distances[:, 29])
            horizon40s += torch.sum(distances[:, 39])

            # Draw sample: the first of the batch
            # if loader == self.test_loader:
            #     self.draw_track(past[0],
            #                     future[0],
            #                     preds[0],
            #                     index_tracklet=step,
            #                     num_epoch=epoch,
            #                     train=False
            #                     )

        dict_metrics['eucl_mean'] = eucl_mean / len(loader.dataset)
        dict_metrics['horizon10s'] = horizon10s / len(loader.dataset)
        dict_metrics['horizon20s'] = horizon20s / len(loader.dataset)
        dict_metrics['horizon30s'] = horizon30s / len(loader.dataset)
        dict_metrics['horizon40s'] = horizon40s / len(loader.dataset)

        dict_metrics['eucl_mean'] = eucl_mean / len(loader.dataset)
        dict_metrics['ADE_1s'] = ADE_1s / len(loader.dataset)
        dict_metrics['ADE_2s'] = ADE_2s / len(loader.dataset)
        dict_metrics['ADE_3s'] = ADE_3s / len(loader.dataset)

        return dict_metrics

    def generate_square_subsequent_mask(self,sz):
        mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _train_single_epoch(self):
        """
        Training loop over the dataset for an epoch
        :return: loss
        """
        config = self.config
        for step, (index, past, future, presents, angle_presents, videos, vehicles, number_vec, scene, scene_one_hot) \
                in enumerate(tqdm.tqdm(self.train_loader)):
            self.iterations += 1
            past = Variable(past)
            future = Variable(future)
            if config.cuda:
                past = past.cuda()
                future = future.cuda()
            self.opt.zero_grad()

            # Get prediction and compute loss
            src_seq_len = past.shape[1] + future.shape[1]
            output = self.mem_n2n(past, future.repeat_interleave(self.config.num_prediction, dim=2))
            # output = self.mem_n2n(past, future)
            future_repeat = future.unsqueeze(1).repeat(1, self.config.num_prediction, 1, 1)
            distances = torch.norm(output - future_repeat, dim=3)
            distances_mean = torch.mean(distances, dim=2)
            index_min = torch.argmin(distances_mean, dim=1)
            best_pred = output[torch.arange(past.shape[0]), index_min[:]]
            loss = self.criterionLoss(best_pred, future)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.mem_n2n.parameters(), 1.0, norm_type=2)
            self.opt.step()

            # Tensorboard summary: loss
            self.writer.add_scalar('loss/loss_total', loss, self.iterations)

        return loss.item()

    def save_results(self, dict_metrics_test, epoch=0):
        """
        Serialize results
        :param dict_metrics_test: dictionary with test metrics
        :param epoch: epoch index (default: 0)
        :return: None
        """
        self.file = open(self.folder_test + "results_"+str(epoch)+".txt", "w")
        self.file.write("TEST:" + '\n')
        self.file.write("split test: " + str(self.data_test.ids_split_test) + '\n')
        # self.file.write("num_predictions:" + str(self.config.preds) + '\n')
        self.file.write("epoch: " + str(epoch) + '\n')
        self.file.write("TRAIN size: " + str(len(self.data_train)) + '\n')
        self.file.write("TEST size: " + str(len(self.data_test)) + '\n')
        # self.file.write("memory size: " + str(len(self.mem_n2n.memory_past)) + '\n')

        self.file.write("error 1s: " + str(dict_metrics_test['horizon10s'].item()) + '\n')
        self.file.write("error 2s: " + str(dict_metrics_test['horizon20s'].item()) + '\n')
        self.file.write("error 3s: " + str(dict_metrics_test['horizon30s'].item()) + '\n')
        self.file.write("error 4s: " + str(dict_metrics_test['horizon40s'].item()) + '\n')
        self.file.write("ADE 1s: " + str(dict_metrics_test['ADE_1s'].item()) + '\n')
        self.file.write("ADE 2s: " + str(dict_metrics_test['ADE_2s'].item()) + '\n')
        self.file.write("ADE 3s: " + str(dict_metrics_test['ADE_3s'].item()) + '\n')
        self.file.write("ADE 4s: " + str(dict_metrics_test['eucl_mean'].item()) + '\n')

        self.file.close()
