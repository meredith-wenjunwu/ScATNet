import tqdm
import torch
from utilities.print_fc import print_report
from sklearn.metrics import accuracy_score
import os
from torch import nn
from experiment.EMA import EMA
from utilities.visualize import compute_stats, compute_case, visualize_top_k_crop
import numpy as np
import pdb


class experiment_engine(object):
    def __init__(self, train_loader,
                 val_loader, train_val_loader, test_loader, **args):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_val_loader = train_val_loader
        self.test_loader = test_loader
        self.__dict__.update(**args)
        # moving average of all
        self.EMA1 = None

    def multi_scale_features(self, multi_data, mask, labels_conf, feature_extractor):
        multi_feat = []
        num_scales = len(multi_data)
        for j in range(num_scales):
            _data = multi_data[j]
            if feature_extractor is not None:
                split_size = self.max_bsz_cnn_gpu0 if self.max_bsz_cnn_gpu0 > 0 else len(_data)
                split_data = torch.split(_data, split_size_or_sections=split_size, dim=1)

                feat = []
                for data in split_data:
                    bsz, n_split, channels, width, height = data.size()
                    if self.use_gpu:
                        data = data.to(device=self.gpu_id[0])
                    # reshape
                    data = data.contiguous().view(bsz * n_split, channels, width, height)

                    if self.finetune_base_extractor and self.training:
                        data = feature_extractor(data)
                        data = data.contiguous().view(bsz, n_split, -1)
                    else:
                        feature_extractor.eval()
                        with torch.no_grad():
                            data = feature_extractor(data)
                            data = data.contiguous().view(bsz, n_split, -1)

                    feat.append(data)

                feat = torch.cat(feat, dim=1)
                multi_feat.append(feat)

            if self.use_gpu:
                if self.mask_type == 'return-indices' and self.mask is not None:
                    mask = [m.to(self.gpu_id[0]) for m in mask]
                else:
                    mask = None
                labels_conf = labels_conf.to(self.gpu_id[0])
        return multi_feat, mask, labels_conf

    def train(self, model, epochs, criterion,
              optimizer, scheduler=None,
              start_epoch=0, feature_extractor=None, **kwargs):

        model.train()
        val_loss_min = 10000
        val_acc_max = -1
        step = 0
        self.EMA1 = EMA(model, ema_momentum=0.001)

        if kwargs['mode'] == 'train':
            dataloader = self.train_loader
        elif kwargs['mode'] == 'merge-train-valid':
            dataloader = self.train_val_loader

        for epoch in range(start_epoch, epochs):
            model.train()
            if self.visdom:
                self.logger.update(epoch, optimizer.param_groups[0]['lr'], mode='lr')
            epoch_loss = 0
            output = []
            scores = []
            target = []
            optimizer.zero_grad()
            for i, (multi_data, labels, labels_conf, paths, mask) in tqdm.tqdm(enumerate(dataloader), leave=False, total=len(dataloader)):
                step += 1
                if self.warmup and step < 500:
                    lr_scale = min(1., float(step + 1) / 500.)
                    for pg in optimizer.param_groups:
                        pg['lr'] = lr_scale * self.lr
                elif step > 500 and self.scheduler == 'cosine':
                    scheduler.step(epoch)
                elif self.scheduler == 'cycle':
                    scheduler.step()
                else:
                    scheduler.step(epoch)
                target.extend(l.item() for l in labels)
                if multi_data[0].dim() != 3:
                    # [B x C x 3 x H x W] x Scales
                    multi_feat, mask, labels_conf = self.multi_scale_features(multi_data=multi_data,
                                                                          feature_extractor=feature_extractor,
                                                                          mask=mask, labels_conf=labels_conf)
                else:
                    # [B x C x F] x Scales
                    labels_conf = labels_conf.to(device=self.gpu_id[0])
                    multi_feat = [d.to(device=self.gpu_id[0]) for d in multi_data]
                    mask=None
                out, _, _ = model(x=multi_feat, src_mask=mask)
                loss = criterion(out, labels_conf)
                loss.backward()
                if (i + 1) % self.aggregate_batch == 0 or (i + 1) == len(dataloader):
                    optimizer.step()
                    self.EMA1.update_parameters(model)
                    optimizer.zero_grad()
                scores.append(out.detach().cpu())
                epoch_loss += loss.detach().cpu().item()
                _, predicted = torch.max(out.data, 1)
                output.extend(predicted.detach().cpu().tolist())
                if self.visdom:
                    self.confusion_meter.add(predicted, labels)
            if self.visdom:
                self.logger.update(epoch, epoch_loss/len(dataloader), mode='train_loss')
                train_acc = accuracy_score(target, output)
                self.logger.update(epoch, train_acc, mode='train_err')
                self.logger.update(epoch, self.confusion_meter.value(), mode='train_mat')

            else:
                print('loss: {:0.23f}'.format(epoch_loss/len(dataloader)))
                print_report(output, target, name='Train', epoch=epoch)

            val_loss, val_acc, summary = self.eval(model, criterion,
                                 epoch=epoch, mode='merge-train-valid', feature_extractor=feature_extractor)
            if step > 500 and self.scheduler == 'reduce':
                scheduler.step(val_loss)
            if val_acc >= val_acc_max:
                # self.eval(model, criterion,
                #           epoch=epoch, mode='test', feature_extractor=feature_extractor)
                print('Valid acc increased ({:.6f} --> {:.6f}).  Saving model...'.format(val_acc_max, val_acc))
            self.save_model(model, epoch, val_acc)
            if val_acc > val_acc_max:
                val_acc_max = val_acc
            elif val_loss < val_loss_min:
                val_loss_min = val_loss

        self.logger.close()




    def save_model(self, model, epoch, loss):
        from pathlib import Path
        if epoch == 'EMA':
            d = os.path.join(self.model_dir, 'EMA')
        else:
            d = self.model_dir
        save_path = os.path.join(d, '{}_{}_{}.pt'.format(self.save_name, epoch, loss))


        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)


    def calc_gradient(self, model, data, mask, save_name):
        # batch_size in data must be one
        # save_name should be a string
        if len(data.shape) < 4:
            data.unsqueeze(0)
        if len(mask.shape) == 1:
            mask=None
        elif len(mask.shape) < 4:
            mask.unsqueeze(0)
        data.requires_grad = True
        with torch.enable_grad():
            output,_ = model({'image': data, 'mask': mask, 'save_name':save_name})
            grad = torch.autograd.grad(output.max(1)[0],
                                       data,
                                       only_inputs=True,
                                       allow_unused=False, create_graph=False)
            # BCHW --> CHW
            grad = grad[0].squeeze(0)
            # CHW --> HW
            #min_val = torch.min(grad)
            #grad += abs(min_val)
            grad = grad ** 2
            grad = torch.mean(grad, dim=0)
            grad = torch.sqrt(grad)
            # min-max normalization
            min_val = torch.min(grad)
            max_val = torch.max(grad)
            grad = torch.add(grad, -min_val)
            grad = torch.div(grad, max_val - min_val)
            grad *= 255.0
            grad = grad.byte().cpu().numpy()


        _, predicted = torch.max(output.data, 1)
        result = predicted.detach().cpu().item()
        return grad, output, result


    def eval(self, model, criterion,
             epoch=None, mode='val', feature_extractor=None):

        return self.eval_crop(model, criterion, epoch=epoch,
                              mode=mode, feature_extractor=feature_extractor)

    def eval_crop(self, model, criterion,
                  epoch=None, mode='val', feature_extractor=None):
        if 'merge-train-valid' in mode:
            dataloader = self.train_val_loader
        elif 'val' in mode or 'ema' in mode:
            dataloader = self.val_loader
        elif 'test' in mode:
            dataloader = self.test_loader
        elif 'train' in mode:
            dataloader = self.train_loader
        else:
            import sys
            sys.exit('Wrong evaluation mode. Choices are val or test')

        model.eval()
        val_target = []
        val_output = []
        val_loss = 0
        scores = []
        results_list = []
        if self.visdom:
            self.confusion_meter.reset()
        with torch.no_grad():
            for i, (multi_data, target, target_conf, paths, mask) in tqdm.tqdm(enumerate(dataloader),
                                                                               leave=False,
                                                                               total=max(1, len(dataloader))):
                val_target.extend(t.item() for t in target)
                labels_conf = target_conf.to(device=self.gpu_id[0])

                if multi_data[0].dim() != 3:
                    # [B x C x 3 x H x W] x Scales
                    multi_feat, mask, labels_conf = self.multi_scale_features(multi_data=multi_data,
                                                                          feature_extractor=feature_extractor,
                                                                          mask=mask, labels_conf=labels_conf)
                else:
                    # [B x C x F] x Scales
                    multi_feat = [d.to(device=self.gpu_id[0]) for d in multi_data]
                    mask = None

                output, output_vis, scale_attn = model(x=multi_feat, src_mask=mask)

                if self.loss_function != 'bce':
                    probabilities = nn.Softmax(dim=-1)(output.detach().cpu())
                else:
                    probabilities = nn.Sigmoid()(output.detach().cpu())
                scores.append(output.detach().cpu())
                val_loss += criterion(output, labels_conf).detach().cpu().item()
                _, predicted = torch.max(output.data, 1)
                val_output.extend(predicted.detach().cpu().tolist())
                for j in range(len(paths)):
                    results_list.append((paths[j], int(target[j].item()), int(predicted[j].item())))
                if self.save_top_k > 0:
                    visualize_top_k_crop(multi_data, self.save_top_k,
                                         predicted, target, output_vis,
                                         paths, self.savedir,
                                         multi_scale=self.multi_scale >= 1,
                                         mask=mask)
                if self.save_result:
                    for j in range(len(paths)):
                        with open(os.path.join(self.savedir, '{}_result.txt'.format(mode)), 'a') as f:
                            f.write('{}; {}; {}; {}\n'.format(paths[j], int(predicted[j].item()),
                                                          probabilities[j].tolist(), ''))

                if self.visdom:
                    self.confusion_meter.add(predicted, target)
        scores = torch.cat(scores, dim=0)
        scores = scores.float()
        if self.loss_function == 'bce':
            predictions_max_sm = nn.Sigmoid()(scores)
        else:
            predictions_max_sm = nn.Softmax(dim=-1)(scores)  # Image x Classes
        pred_label_max1 = torch.max(predictions_max_sm, dim=-1)[1]  # Image x 1
        pred_label_max = pred_label_max1.byte().cpu().numpy().tolist()  # Image x 1
        pred_conf_max = predictions_max_sm.float().cpu().numpy()  # Image x Classes
        # val_acc = accuracy_score(val_target, val_output)
        
        val_acc, results_summary = compute_case(results_list, pred_conf_max, verbose=(epoch is None), mode=mode,
                                   savepath=self.savedir, save=self.save_result)
        if self.mode == 'test' or self.mode == 'valid':
            sp = self.savedir
        else:
            sp = None
        if self.visdom:
            self.logger.update(epoch, val_loss / max(1,len(dataloader)), mode='{}_loss'.format(mode))
            self.logger.update(epoch, val_acc, mode='{}_err'.format(mode))
            self.logger.update(epoch, self.confusion_meter.value(), mode='{}_mat'.format(mode))
        else:
            compute_stats(y_true=val_target, y_pred=pred_label_max, y_prob=pred_conf_max, logger=None,
                          mode='{}_roc'.format(mode), num_classes=self.num_classes,
                          savepath=sp, fname=self.save_name)
            name = 'Valid' if mode == 'val' else 'Test'
            print_report(val_output, val_target, name=name, epoch=epoch)
        return val_loss / max(1, len(dataloader)), val_acc, results_summary


