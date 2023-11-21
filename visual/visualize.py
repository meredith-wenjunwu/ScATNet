import os
import pdb
import torch
from torch import nn
import cv2
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
from numpy import interp
import seaborn as sns
import cmapy
from utilities.util import save_metrics
from visual.color_encoder import ColorEncoder
from sklearn.metrics import classification_report, confusion_matrix
from metrics.cmat_metrics import CMMetrics, CMResults

'''
This file defines functions for plotting ROC curves and gradients. s
'''

rcParams['font.family'] = 'monospace'
rcParams['font.size'] = 12

font_main_axis = {
    'weight': 'bold',
    'size': 12
}

LINE_WIDTH = 1.5

MICRO_COLOR = 'k'  # (255/255.0, 127/255.0, 0/255.0)
MACRO_COLOR = 'k'  # (255/255.0,255/255.0,51/255.0)
MICRO_LINE_STYLE = 'dashed'
MACRO_LINE_STYLE = 'solid'

CLASS_LINE_WIDTH = 2

GRID_COLOR = (204 / 255.0, 204 / 255.0, 204 / 255.0)
GRID_LINE_WIDTH = 0.25
GRID_LINE_STYLE = ':'


def visual_save_grad(predicted, label, grads,
                     images, savepath,
                     img_paths):
    for i in range(len(grads)):
        im_h, im_w = grads[i].shape[:2]
        grad = grads[i] if grads is not None else None
        image = np.array(images[i])
        # if predicted[0].item() == label[0].item():
        #     dir = 'correct'
        # else:
        #     dir = 'wrong'
        dir = 'all'
        im_path = img_paths[0]
        basename = os.path.splitext(os.path.basename(im_path))[0]
        im_dir = os.path.join(savepath, 'visual', dir, str(label[0].item()), basename)
        scale_dir = os.path.join(im_dir, 'scale_' + str(i))
        if not os.path.exists(scale_dir):
            os.makedirs(scale_dir)
        # grad = cv2.cvtColor(grad, cv2.COLOR_RGB2BGR)
        gradient_heat = cv2.applyColorMap(grad, cv2.COLORMAP_INFERNO)
        cv2.imwrite(os.path.join(scale_dir, 'gradient_pixel.jpg'), gradient_heat)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.addWeighted(gradient_heat, 0.6, image, 0.4, 0)
        cv2.imwrite(os.path.join(scale_dir, 'gradient_pixel_overlay.jpg'), image)


def visual_save_attn(attn_wts, savepath, slice_path):
    '''
    attn_wts: num_scales x layers x heads x P
    '''
    image = cv2.imread(slice_path)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    num_patch = int(np.sqrt(attn_wts.shape[-1]))
    for scale in range(attn_wts.shape[0]):
        for layer in range(attn_wts.shape[1]):
            for head in range(attn_wts.shape[2]):
                out_dir = os.path.join(savepath, 'scale_'+str(scale)+'_layer_'+str(layer), 'head_'+str(head))
                os.makedirs(out_dir, exist_ok=True)
                attn_grid = attn_wts[scale, layer, head].reshape(num_patch, num_patch)
                torch.save(attn_grid, os.path.join(out_dir, '{}_patch_attn.pth'.format(os.path.basename(slice_path).replace('.tif', ''))))
                overlayed = overlay_attn(attn_grid, image)
                cv2.imwrite(os.path.join(out_dir, '{}_attn_overlay.jpg'.format(os.path.basename(slice_path).replace('.tif', ''))), overlayed)


def overlay_attn(attn_map, image):
    attn_grid = (255 * attn_map).astype(np.uint8)
    attn_grid_img = cv2.applyColorMap(cv2.resize(attn_grid, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST), cv2.COLORMAP_INFERNO)
    overlayed = cv2.addWeighted(attn_grid_img, 0.6, image, 0.4, 0)
    return overlayed


def visualize_layer(x, size=(100, 100)):
    with torch.no_grad():
        assert x.dim() == 4
        y = nn.Upsample(size=size, mode='bilinear')(x)
        y = nn.Softmax(dim=1)(y)
        y = torch.max(y, dim=1)[0] # torch.max --> prob, indexes
        #y /= (torch.max(y) + 1e-7)
        y *= 255.0
        y = y.byte().cpu().numpy()
        y_b = y[0]
            #print(y_b)
            #print(y_b)
            #y_b = Image.fromarray(y_b)
            #plt.imshow(y_b, cmap='hot', interpolation='nearest')
            #plt.savefig('../vis/{}_{}.jpg'.format(name, b_ind))

        #colormap = plt.get_cmap('inferno')
        #y_heat = (colormap(y_b) * 2**16).astype(np.uint16)[:,:,:3]
        #y_heat = cv2.cvtColor(y_heat, cv2.COLOR_RGB2BGR)

        y_heat = cv2.applyColorMap(y_b, cmapy.cmap('inferno'))
    return y_heat



def save_attn_weights_grads(predicted, label, out_vis, scale_attn, savepath, img_paths, grads=None):
    for i in range(len(out_vis)):
        bsz, n_crops, _ = out_vis[i].size()
        for b_i in range(bsz):
            grad = grads[i][b_i] if grads is not None else None
            vis = np.array(out_vis[i][b_i].cpu())
            # if predicted[b_i].item() == label[b_i]:
            #     dir = 'correct'
            # else:
            #     dir = 'wrong'
            dir = 'all'
            im_path = img_paths[b_i]
            basename = os.path.splitext(os.path.basename(im_path))[0]
            im_dir = os.path.join(savepath, 'visual' ,dir, str(label[b_i].cpu().item()), basename)
            scale_dir = os.path.join(im_dir, 'scale_' + str(i))
            if not os.path.exists(scale_dir):
                os.makedirs(scale_dir)
            torch.save(vis, os.path.join(scale_dir, 'patch_attn.pth'))
            if grad is not None:
                torch.save(np.array(grad), os.path.join(scale_dir, 'patch_grads.pth'))
            # plot patch wise heatmap
            fig = plt.figure()
            ax = sns.heatmap(vis, linewidth=0.5)
            fig.savefig(os.path.join(scale_dir, 'patch_attn.jpg'))
            if scale_attn is not None and i == 0:
                s_attn = np.array(scale_attn[b_i].cpu())
                torch.save(s_attn, os.path.join(im_dir, 'scale_attn.pth'))
                plt.clf()
                fig = plt.figure()
                ax = sns.heatmap(s_attn, linewidth=0.5)
                fig.savefig(os.path.join(im_dir, 'scale_attn.jpg'))
            plt.close('all')



def visualize_top_k_crop(crops, k, predicted, label, out_vis, img_paths,
                         savepath, multi_scale=False, mask=None):
    from dataset.transforms import NormalizeInverse
    from torchvision.utils import save_image
    # if not multi_scale:
    #     crops = [crops]
    #     out_vis = [out_vis]
    unorm = NormalizeInverse(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    for i in range(len(crops)):
        crop = crops[i]
        bsz, n_crops, channels, width, height = crop.size()
        # sort out_vis
        for b_i in range(bsz):
            vis = out_vis[i]
            if predicted[b_i].item() == label[b_i]:
                continue
            _, indices = torch.sort(vis, dim=1, descending=True)
            top_k_indices = indices[:, :k].cpu().squeeze(2)
            batch = crop[b_i]
            crops_selected = torch.index_select(batch, dim=0, index=top_k_indices[b_i])

            im_path = img_paths[b_i]
            basename = os.path.splitext(os.path.basename(im_path))[0]
            im_dir = os.path.join(savepath, str(label[b_i].cpu().item()), basename)
            scale_dir = os.path.join(im_dir, str(i))

            if mask is not None:
                mask_bi = mask[i][b_i]
                indices = np.nonzero(mask_bi)
                mask_dir = os.path.join(savepath, str(label[b_i].cpu().item()), basename, 'mask')
                if not os.path.exists(mask_dir):
                    os.makedirs(mask_dir)
                for d in indices:
                    outpath = os.path.join(mask_dir, '{}.jpg'.format(d.item()))
                    mask_selected = torch.index_select(batch, dim=0, index=d.cpu())
                    save_image(unorm(mask_selected.squeeze(0)), outpath)
            # unorm crop
            # save crops in save path
            # save sorted indices
            if not os.path.exists(im_dir):
                os.makedirs(im_dir)
            if not os.path.exists(scale_dir):
                os.makedirs(scale_dir)
            with open(os.path.join(scale_dir, 'indices.txt'), 'a') as f:
                f.write('{}\n'.format(im_path))
                for ind in top_k_indices[b_i]:
                    f.write('{}\n'.format(ind.item()))
            for d in range(k):
                outpath = os.path.join(scale_dir, '{}.jpg'.format(d))
                save_image(unorm(crops_selected[d]), outpath)



def compute_stats(y_true, y_pred, y_prob, logger, mode, num_classes, savepath=None, fname='results.json'):
    results_summary = dict()
    # Add true labels and predictions
    results_summary['true_labels'] = [int(x) for x in y_true]
    results_summary['pred_labels'] = [int(x) for x in y_pred]
    #print(len(y_true))
    #print(len(y_pred))
    cmat = confusion_matrix(y_true, y_pred)
    cmat_np_arr = np.array(cmat)


    # Compute results from confusion matrix
    conf_mat_eval = CMMetrics()
    cmat_results: CMResults = conf_mat_eval.compute_metrics(conf_mat=cmat_np_arr)

    cmat_results_dict = cmat_results._asdict()
    for key, values in cmat_results_dict.items():
        if isinstance(values, np.ndarray):
            values = values.tolist()
        results_summary['{}'.format(key)] = values

    if savepath is not None:
        save_metrics(results_summary, '{}/{}_summary'.format(savepath, fname))

    # plot the ROC curves
    y_true = np.array(y_true, dtype=int)
    y_true_oh = np.eye(num_classes)[y_true]
    if y_prob is not None:
        y_prob = np.array(y_prob)

        plot_roc(
            ground_truth=y_true_oh,
            pred_probs=y_prob,
            n_classes=num_classes,
            logger=logger, mode=mode,
            savepath=savepath, fname=fname)
    return results_summary


def compute_case(results_list, scores, verbose=False, savepath=None, mode=None, save=False):
    # pred_label_max1 = torch.max(scores, dim=-1)[1]
    # pred_label_max = pred_label_max1.byte().cpu().numpy().tolist()  # Image x 1
    # scores = scores.float().cpu().numpy()  # Image x Classes

    case_pred = {}
    case_target = {}

    for i, line in enumerate(results_list):
        im_p = line[0]
        label = int(line[1])
        pred = int(line[2])
        bn = os.path.basename(im_p)
        im_ind = os.path.splitext(bn)[0]
        case = im_ind.split('_')
        case = case[0] + '_' + case[1]
        if case not in case_pred:
            case_pred[case] = (pred, scores[i,:])
            case_target[case] = label
        else:
            if pred > case_pred[case][0]:
                case_pred[case] = (pred, scores[i, :])
            if pred == case_pred[case][0]:
                # check which score is higher
                s1 = case_pred[case][1]
                s2 = scores[i, :]
                if s1[pred] < s2[pred]:
                    case_pred[case] = (pred, s2)

    case_confusion_matrix = np.zeros((4, 4))
    pred_list = []
    label_list = []
    y_prob = []

    for case in case_pred:
        pred, score = case_pred[case]
        label = case_target[case]
        case_confusion_matrix[label][pred] += 1
        pred_list.append(pred)
        label_list.append(label)
        y_prob.append(score)
    classification_report(label_list, pred_list, digits=4)
    results_summary = dict()
    results_summary['true_labels'] = [int(x) for x in label_list]
    results_summary['pred_labels'] = [int(x) for x in pred_list]
    # print(len(y_true))
    # print(len(y_pred))
    cmat = confusion_matrix(results_summary['true_labels'], results_summary['pred_labels'])
    cmat_np_arr = np.array(cmat)
    conf_mat_eval = CMMetrics()
    cmat_results: CMResults = conf_mat_eval.compute_metrics(conf_mat=cmat_np_arr)
    cmat_results_dict = cmat_results._asdict()
    for k, v in cmat_results_dict.items():
        if isinstance(v, np.ndarray):
            v = v.tolist()
        results_summary['{}'.format(k)] = v
    if verbose:
        print('-------------Case Report----------------')
        print('Case Overall Accuracy: {0:0.4f}'.format(results_summary['overall_accuracy']))
        print(case_confusion_matrix)
        print('Case Specificity: {0:0.4f}'.format(results_summary['specificity_micro']))
        class_acc = case_confusion_matrix.diagonal() / case_confusion_matrix.sum(axis=1)
        print('Class 1&2 \t Class 3 \t Class 4 \t Class 5')
        print('{0:0.4f} \t \t {1:0.4f} \t \t {2:0.4f} \t \t {3:0.4f}'.format(class_acc[0], class_acc[1], class_acc[2], class_acc[3]))

    # plot the ROC curves
    y_true = np.array(results_summary['true_labels'].copy())
    num_classes = np.max(y_true) + 1
    y_true_oh = np.eye(num_classes)[y_true]
    if y_prob is not None:
        y_prob = np.array(y_prob)
    if savepath is not None and save:
        save_metrics(results_summary, '{}/{}_case_summary'.format(savepath, mode))

        plot_roc(
            ground_truth=y_true_oh,
            pred_probs=y_prob,
            n_classes=num_classes,
            logger=None, mode=mode,
            savepath=savepath, fname='case_level')
    return results_summary['overall_accuracy'], results_summary



def plot_roc(ground_truth, pred_probs, n_classes, logger, mode,
             class_names=None, dataset_name='bbwsi',
             savepath=None, fname='roc_curve'):
    from sklearn.metrics import roc_curve, auc

    class_colors, class_linestyles = ColorEncoder().get_colors(dataset_name=dataset_name)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    # compute ROC curve class-wise
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(ground_truth[:, i], pred_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # COMPUTE MICRO-AVERAGE ROC CURVE AND ROC AREA
    fpr["micro"], tpr["micro"], _ = roc_curve(ground_truth.ravel(), pred_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # COMPUTE MACRO-AVERAGE ROC CURVE AND ROC AREA

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # PLOT the curves
    micro_label = 'Micro avg. (AUC={0:0.2f})'.format(roc_auc["micro"])
    plt.plot(fpr["micro"], tpr["micro"], label=micro_label, color=MICRO_COLOR,
             linestyle=MICRO_LINE_STYLE, linewidth=LINE_WIDTH)

    macro_label = 'Macro avg. (AUC={0:0.2f})'.format(roc_auc["macro"])
    plt.plot(fpr["macro"], tpr["macro"], label=macro_label, color=MACRO_COLOR,
             linestyle=MACRO_LINE_STYLE, linewidth=LINE_WIDTH)
    class_names =['MMD', 'MIS', 'pT1a', 'pT1b']
    # pdb.set_trace()
    if class_names is not None:
        assert len(class_names) == n_classes
        for i, c_name in enumerate(class_names):
            label = "{0} (AUC={1:0.2f})".format(c_name, roc_auc[i])
            plt.plot(fpr[i], tpr[i], color=class_colors[i],
                     lw=CLASS_LINE_WIDTH, label=label, linestyle=class_linestyles[i])
    else:
        for i, color in zip(range(n_classes), class_colors):
            label = 'Class {0} (AUC={1:0.2f})'.format(i, roc_auc[i])
            plt.plot(fpr[i], tpr[i], color=color, lw=CLASS_LINE_WIDTH,
                     label=label, linestyle=class_linestyles[i])

    plt.plot([0, 1], [0, 1], 'tab:gray', linestyle='--', linewidth=1)
    # plt.grid(color=GRID_COLOR, linestyle=GRID_LINE_STYLE, linewidth=GRID_LINE_WIDTH)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontdict=font_main_axis)
    plt.ylabel('True Positive Rate', fontdict=font_main_axis)
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.legend(edgecolor='black', loc="best")
    # plt.title('{} Curve'.format(mode))
    if logger is not None:
        logger.update(plt, mode=mode)
    # plt.tight_layout()
    if savepath is not None:
        plt.savefig('{}/{}_{}_roc.pdf'.format(savepath, mode, fname), dpi=300, bbox_inches='tight')
    plt.close()


