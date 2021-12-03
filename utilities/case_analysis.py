from os import path
import numpy as np
import argparse
from os.path import join, split
from sklearn.metrics import classification_report
from metrics.cmat_metrics import CMMetrics, CMResults



def open_txt(results_fn):
    with open(results_fn, 'r') as f:
        lines = [line.rstrip() for line in f]
    return lines


def parse_matrix(raw):
    confusion_matrix = np.zeros((4, 4), dtype=int)
    confidence_matrix = {0: {}, 1: {}, 2: {}, 3: {}}
    confidence_case_matrix = {}
    case_pred = {}
    case_target = {}
    case_lookup = {0: {}, 1: {}, 2: {}, 3: {}}

    for line in raw:
        line = line.split(',')
        im_p = line[0]
        pred = int(line[1])
        #     scores_raw = line[2].split(',')
        #     scores = [float(scores_raw[i].replace('[', '').replace(']', '')) for i in range(len(scores_raw))]
        scores_raw = line[2:2 + 4]
        scores = [float(scores_raw[i].replace('[', '').replace(']', '')) for i in range(len(scores_raw))]
        label = max(0, int(path.basename(path.dirname(path.dirname(im_p)))) - 2)
        bn = path.basename(im_p)
        im_ind = path.splitext(bn)[0]
        case = im_ind.split('_x10')[0]
        confusion_matrix[label][pred] += 1
        if pred not in confidence_matrix[label]:
            confidence_matrix[label][pred] = [scores]
            case_lookup[label][pred] = [im_ind]
        else:
            confidence_matrix[label][pred].append(scores)
            case_lookup[label][pred].append(im_ind)
        if case not in confidence_case_matrix:
            confidence_case_matrix[case] = {}
            case_pred[case] = pred
            case_target[case] = label
        else:
            if pred > case_pred[case]:
                case_pred[case] = pred
        confidence_case_matrix[case][str(im_ind[-1])] = scores
    return confusion_matrix, confidence_matrix, confidence_case_matrix, case_pred, case_target



def print_case_report(case_pred, case_target):
    case_confusion_matrix = np.zeros((4, 4))
    pred_list = []
    label_list = []
    for case in case_pred:
        pred = case_pred[case]
        label = case_target[case]
        case_confusion_matrix[label][pred] += 1
        pred_list.append(pred)
        label_list.append(label)
    classification_report(label_list, pred_list)
    return pred_list, label_list


def generate_excel(results_dir, confidence_case_matrix, case_pred):
    import csv
    csv_filename = path.join(results_dir, 'case.csv')
    fields = ['case ID', 'Prediction', 'slice number', '0', '1', '2', '3']
    with open(csv_filename, 'w') as csv_f:
        csvwriter = csv.writer(csv_f)
        csvwriter.writerow(fields)
        for case in confidence_case_matrix:
            row = [case]
            row.append(str(case_pred[case]))
            csvwriter.writerow(row)
            for i in confidence_case_matrix[case]:
                slice_row = ['', '']
                slice_row.append(str(i))
                for conf in confidence_case_matrix[case][str(i)]:
                    slice_row.append(str(conf))
                csvwriter.writerow(slice_row)


def main():
    parser = argparse.ArgumentParser(description='Generate case based performance metrics')
    parser.add_argument('--results_dir', required=True, type=str)
    parser.add_argument('--results_fn', required=False, type=str, default='result.txt')
    args = parser.parse_args()
    results_dir = args.results_dir
    results_fn = args.results_fn
    fn = join(results_dir, results_fn)
    raw = open_txt(fn)
    confusion_matrix, confidence_matrix, confidence_case_matrix, case_pred, case_target, case_lookup = parse_matrix(raw)
    print_case_report(case_pred, case_target)
    generate_excel(results_dir, confidence_case_matrix, case_pred)


