import tqdm
import torch
from utilities.print_fc import print_report
from visual.visualize import visual_save_grad, compute_stats, visualize_top_k_crop, compute_case
from sklearn.metrics import accuracy_score
import os
from torch import nn
from experiment.EMA import EMA
import numpy as np
import pdb
from dataset.dataloaders import sliding_dataloader

torch.autograd.set_detect_anomaly(True)


class experiment_engine(object):
    def __init__(self, train_loader,
                 val_loader, test_loader, **args):
        self.train_loader = train_loader
        self.val_loader = val_loader
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
                split_size = self.train_split if self.train_split > 0 else len(_data)
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
              start_epoch=0, feature_extractor=None,):

        model.train()
        val_loss_min = 10000
        val_acc_max = -1
        step = 0
        self.EMA1 = EMA(model, ema_momentum=0.001)
        for epoch in range(start_epoch, epochs):
            model.train()
            if self.visdom:
                self.logger.update(epoch, optimizer.param_groups[0]['lr'], mode='lr')
            epoch_loss = 0
            output = []
            scores = []
            target = []
            optimizer.zero_grad()
            for i, (multi_data, labels, labels_conf, paths, mask) in tqdm.tqdm(enumerate(self.train_loader), leave=False, total=len(self.train_loader)):
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
                out, _, _ = model(multi_feat, src_mask=mask)
                loss = criterion(out, labels_conf)
                loss.backward()
                if (i + 1) % self.aggregate_batch == 0 or (i + 1) == len(self.train_loader):
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
                self.logger.update(epoch, epoch_loss/len(self.train_loader), mode='train_loss')
                train_acc = accuracy_score(target, output)
                self.logger.update(epoch, train_acc, mode='train_err')
                self.logger.update(epoch, self.confusion_meter.value(), mode='train_mat')
                # scores = torch.cat(scores, dim=0)
                # predictions_max_sm = nn.Softmax(dim=-1)(scores)  # Image x Classes
                # pred_label_max = torch.max(predictions_max_sm, dim=-1)[1]  # Image x 1
                # pred_label_max = pred_label_max.byte().cpu().numpy().tolist()  # Image x 1
                # pred_conf_max = predictions_max_sm.float().cpu().numpy()  # Image x Classes
                # print(pred_conf_max.shape)
                # compute_stats(y_true=target, y_pred=pred_label_max, y_prob=pred_conf_max, logger=self.logger,
                #                    mode='train_roc', num_classes=self.num_classes)
            else:
                print('loss: {:0.23f}'.format(epoch_loss/len(self.train_loader)))
                print_report(output, target, name='Train', epoch=epoch)

            val_loss, val_acc = self.eval(model, criterion,
                                 epoch=epoch, mode='val', feature_extractor=feature_extractor)
            if step > 500 and self.scheduler == 'reduce':
                scheduler.step(val_loss)
            ema_val_loss, ema_val_acc = self.eval(self.EMA1.ema_model, criterion, epoch=epoch, mode='ema',
                                                  feature_extractor=feature_extractor)
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


            #max_grad = torch.max(grad_inp[:, i, :, :])
            #min_grad = torch.min(grad_inp[:, i, :, :])
            #grad_inp[:, i, :, :] = ((grad_inp[:, i, :, :] - min_grad) /
            #                    (max_grad - min_grad))

        #grad_inp *= 255
        #grad_inp = grad_inp.byte().cpu().numpy()
        _, predicted = torch.max(output.data, 1)
        result = predicted.detach().cpu().item()
        return grad, output, result


    def eval(self, model, criterion,
             epoch=None, mode='val', feature_extractor=None,
             sliding_window=False):
        if sliding_window:
            return self.eval_sliding(model, criterion, epoch=epoch,
                                     mode=mode, feature_extractor=feature_extractor)
        else:
            return self.eval_crop(model, criterion, epoch=epoch,
                                  mode=mode, feature_extractor=feature_extractor)

    def eval_sliding(self, model, criterion,
                     epoch=None,  mode='val', feature_extractor=None):
        if 'val' in mode or 'ema' in mode:
            dataset = self.val_loader
        elif 'test' in mode:
            dataset = self.test_loader
        elif 'train' in mode:
            dataset = self.train_loader
        else:
            import sys
            sys.exit('Wrong evaluation mode. Choices are val or test, got "{}"'.format(mode))
        model.eval()
        val_target = []
        val_output_ensemble = []
        val_loss = []
        scores = []
        scale_attns = []
        if self.visdom:
            self.confusion_meter.reset()
        with torch.no_grad():
            for i, d_set in tqdm.tqdm(enumerate(dataset), leave=False, total=len(dataset)):
                dataloader = sliding_dataloader(dataset, i)
                case_score = []
                case_predicted = []
                case_loss = []
                case_probabilities = []
                case_scale_attn = []
                for i, (multi_data, target, target_conf, paths, mask) in tqdm.tqdm(enumerate(dataloader),
                                                                                 leave=False,
                                                                                 total=len(dataloader)):
                    if i == 0:
                        val_target.extend(t.item() for t in target)

                    multi_feat, mask, labels_conf = self.multi_scale_features(multi_data=multi_data,
                                                                              feature_extractor=feature_extractor,
                                                                              mask=mask, labels_conf=target_conf)

                    output, output_vis, scale_attn= model(x=multi_feat, src_mask=mask)
                    probabilities = nn.Softmax(dim=-1)(output.detach().cpu())
                    scale_attn = scale_attn.detach().cpu()
                    if len(scale_attn.shape) > 2:
                        scale_attn = scale_attn.squeeze(2)[0]
                    case_scale_attn.append(scale_attn.tolist())
                    case_probabilities.append(probabilities.detach().cpu().tolist()[0])
                    case_score.append(output.detach().cpu().tolist()[0])
                    case_loss.append(criterion(output, labels_conf).detach().cpu().item())
                    _, predicted = torch.max(output.data, 1)
                    case_predicted.extend(predicted.detach().cpu().tolist())
                val_loss.append(np.mean(case_loss))
                scores.append(np.mean(case_score, axis=0))
                scale_attns.append(np.mean(case_scale_attn, axis=0))
                max_output = np.max(case_predicted)
                ind_max = np.argmax(case_predicted)
                val_output_ensemble.append(max_output)

                if self.save_result:
                    with open(os.path.join(self.savedir, 'result.txt'), 'a') as f:
                        f.write('{}; {}; {}; {}\n'.format(paths[0], int(max_output),
                                                      case_probabilities[ind_max], list(np.mean(case_scale_attn, axis=0))))

                if self.visdom:
                    self.confusion_meter.add(predicted, target)

        val_loss = np.mean(val_loss)
        scores = np.stack(scores)
        val_output = np.argmax(scores, axis=1)
        val_acc = accuracy_score(val_target, val_output_ensemble)
        if self.mode == 'test' or self.mode == 'valid':
            sp = self.savedir
        else:
            sp = None
        if self.visdom:
            self.logger.update(epoch, val_loss/len(dataloader), mode='{}_loss'.format(mode))
            self.logger.update(epoch, val_acc, mode='{}_err'.format(mode))
            self.logger.update(epoch, self.confusion_meter.value(), mode='{}_mat'.format(mode))
        else:
            compute_stats(y_true=val_target, y_pred=val_output, y_prob=None, logger=None,
                          mode='{}_roc'.format(mode), num_classes=self.num_classes,
                          savepath=sp, fname=self.save_name)
            name = 'Valid' if mode == 'val' else 'Test'
            print_report(val_output, val_target, name=name+'_average', epoch=epoch)
            print_report(val_output_ensemble, val_target, name=name+'_max', epoch=epoch)

        return val_loss/len(dataloader), val_acc


    def eval_crop(self, model, criterion,
                  epoch=None, mode='val', feature_extractor=None):
        if 'val' in mode or 'ema' in mode:
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
        if 'melanoma' in self.dataset:
            results_list = []
        if self.visdom:
            self.confusion_meter.reset()
        with torch.no_grad():
            for i, (multi_data, target, target_conf, paths, mask) in tqdm.tqdm(enumerate(dataloader),
                                                                               leave=False,
                                                                               total=max(1, len(dataloader))):
                val_target.extend(t.item() for t in target)

                if multi_data[0].dim() != 3:
                    # [B x C x 3 x H x W] x Scales
                    multi_feat, mask, labels_conf = self.multi_scale_features(multi_data=multi_data,
                                                                          feature_extractor=feature_extractor,
                                                                          mask=mask, labels_conf=labels_conf)
                else:
                    # [B x C x F] x Scales
                    labels_conf = target_conf.to(device=self.gpu_id[0])
                    multi_feat = [d.to(device=self.gpu_id[0]) for d in multi_data]
                    mask = None

                #multi_feat, mask, labels_conf = self.multi_scale_features(multi_data=multi_data,
                #                                                          feature_extractor=feature_extractor,
                #                                                          mask=mask, labels_conf=target_conf)

                output, output_vis, scale_attn = model(x=multi_feat, src_mask=mask)
                # scale_attn = scale_attn.detach().cpu()
                # if len(scale_attn.shape) > 2:
                #     scale_attn = scale_attn.squeeze(2)
                # scale_attn = scale_attn.tolist()
                if self.loss_function != 'bce':
                    probabilities = nn.Softmax(dim=-1)(output.detach().cpu())
                else:
                    probabilities = nn.Sigmoid()(output.detach().cpu())
                scores.append(output.detach().cpu())
                val_loss += criterion(output, labels_conf).detach().cpu().item()
                _, predicted = torch.max(output.data, 1)
                val_output.extend(predicted.detach().cpu().tolist())
                if 'melanoma' in self.dataset:
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
        val_acc = accuracy_score(val_target, val_output)
        if 'melanoma' in self.dataset:
            val_acc = compute_case(results_list, pred_conf_max, verbose=(epoch is None), mode=mode,
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
        return val_loss / max(1, len(dataloader)), val_acc


