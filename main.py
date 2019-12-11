#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 01:05:00 2019

@author: wuwenjun
"""
from argparse import ArgumentParser



if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--mode', required=True,
                        default='kmeans',
                        const='kmeans',
                        nargs='?',
                        choices=['feature', 'kmeans',
                        'bag_of_words', 'classifier_train', 'classifier_test'],
                        help='Choose mode from k-means clustering, visualization and classification_training, classification_testing')
    parser.add_argument('--trained_kmeans_cluster', default=None,
       help='Previously trained kmeans clusters')
    parser.add_argument('--trained_hclusters', default=None,
        )
    parser.add_argument('--single_image', default=None, help='Input image path')
    parser.add_argument('--single_image_label', default='*.pkl', help="Label for training or testing for single image input")
    parser.add_argument('--image_level', default=0, help="Input Argument for openslide package see https://openslide.org/")
    parser.add_argument('--image_folder', default=None, help='Input image batch folder' )
    parser.add_argument('--image_folder_label', default=None, help='Input image batch folder' )
    parser.add_argument('--image_format', required=True, default='.jpg',
        choices=['.jpg', '.png', '.tif', '.mat', '.tiff'],help='Input format')
    parser.add_argument('--save_intermediate', default=False, help='Whether or not to save the intermediate results')
    parser.add_argument('--dict_size', default= 40, help='Dictionary Size for KMeans')
    parser.add_argument('--histogram_bin', default=64, help='Bin size for histogram')
    parser.add_argument('--save_path', default='/Users/wuwenjun/Documents/UW/Research/ITCR/Poster/output/test', help='save_path for outputs: e.g.features, kmeans, classifier')
    parser.add_argument('--word_size', default=120, help='Size of a word (in a bag of words model)')
    parser.add_argument('--bag_size', default=3600, help="Size of a bag (in a bag of words model)")
    parser.add_argument('--overlap_bag', default=2400, help='Overlapping pixels between bags')
    parser.add_argument('--ROI_csv', default=None, help='Input csv file for ROI tracking data')
    parser.add_argument('--WSI_csv', default=None, help='Input csv file for WSI size by case ID')
    parser.add_argument('--classifier', default='logistic',
                        const='logistic',
                        nargs='?',
                        choices=['logistic', 'svm'])
    parser.add_argument('--trained_model', default=None, help='previously trained model path')
    parser.add_argument('--lr', default=0.001, help='initial learning rate')
    parser.add_argument('--learning_rate', default='optimal', help='https://scikit-learn.org/0.15/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier')
    args = parser.parse_args()

    mode = args.mode
    image_path = args.single_image
    image_label_path = args.single_image_label
    folder_path = args.image_folder
    folder_label_path = args.image_folder_label
    ext = args.image_format
    save_flag = args.save_intermediate
    dict_size = int(args.dict_size)
    histogram_bin = int(args.histogram_bin)
    save_path = args.save_path
    word_size = args.word_size
    roi_csv_file = args.ROI_csv
    wsi_csv_file = args.WSI_csv
    bag_size = int(args.bag_size)
    overlap = int(args.overlap_bag)
    clf_filename = args.trained_model
    kmeans = args.trained_kmeans_cluster
    hcluster = args.trained_hclusters
    image_level = args.image_level

    import os
    os.environ['OPENCV_IO_MAX_IMAGE_PIXELS']=str(2**64)
    import cv2
    import numpy as np
    from cluster import *

    import pickle
    import glob
    from util import *
    from classifier import *
    import openslide
    import sys

    print("==================MODE: {}==================".format(mode))

    if mode == 'kmeans':


        kmeans = pickle.load(open(kmeans, 'rb')) if kmeans is not None else None
        hcluster = pickle.load(open(hcluster, 'rb')) if hcluster is not None else None

        if image_path is not None and save_path is not None:
            # Save features
            filename = save_path + '_feat.pkl'
            result = get_feat_from_image(image_path, save_flag, word_size, filename)

            # K-Means Part

            filename = save_path + '_kmeans.pkl'
            kmeans = construct_kmeans(result) if first_image else kmeans.partial_fit_kmeans(result, kmeans)

            pickle.dump(kmeans, open(filename, 'wb'))

        elif folder_path is not None and save_path is not None:
            print('-------Running Batch Job-------')
            # Feature computation and K-Means clustering in batch

            im_list = sorted(glob.glob(os.path.join(folder_path, '*' + ext)))
            count = 0
            print('# of images: %r' %(len(im_list)))
            # if the image processed is small, combine with the next image in queue
            small_image = False
            for im_p in im_list:
                if count % 10 == 0: 
                    print('Processed %r / %r' %(count, len(im_list)))

                    if count != 0:
                        hcluster = h_cluster(kmeans, final_size=dict_size)
                        filename = os.path.join(save_path, 'kmeans.pkl')
                        filename2 = os.path.join(save_path, 'hcluster.pkl')
                        pickle.dump(kmeans, open(filename, 'wb'))
                        pickle.dump(hcluster, open(filename2, 'wb'))
                count += 1
                # get filename without extension
                base = os.path.basename(im_p)
                path_noextend = os.path.splitext(base)[0]
                fname = path_noextend  + '_feat.pkl'
                if 'part' in path_noextend:
                    alter_fname = path_noextend.split('part')[0] + 'feat.pkl'
                    alter_fname = os.path.join(save_path, alter_fname)
                else:
                    alter_fname = fname
                filename = os.path.join(save_path, fname)
                if small_image: temp = result
                if os.path.exists(filename) or os.path.exists(alter_fname):
                    result = pickle.load(open(filename, 'rb'))
                else:
                    if ext != '.mat':
                        result = get_feat_from_image(im_p, save_flag, word_size, save_path=filename)
                    else:
                        im, m = load_mat(im_p)
                        result = get_feat_from_image(None, save_flag, word_size, image=im)
                        pickle.dump(result, open(filename, 'wb'))
                if small_image:
                    result = np.concatenate((result, temp), axis=0)
                    small_image = False
                if result.shape[0] < 200: small_image=True
                else: small_image = False
                # Online-Kmeans
                if kmeans is None and (not small_image):
                    kmeans = construct_kmeans(result)
                    assert kmeans is not None
                elif kmeans is not None and (not small_image):
                    kmeans = partial_fit_k_means(result, kmeans)
                    assert kmeans is not None, "kmeans construction/update invalid"

            hcluster = h_cluster(kmeans, final_size=dict_size)
            filename = os.path.join(save_path, 'kmeans.pkl')
            filename2 = os.path.join(save_path, 'hcluster.pkl')
            pickle.dump(kmeans, open(filename, 'wb'))
            pickle.dump(hcluster, open(filename2, 'wb'))

        else:
            print('Error: Input image path is None or save path is None.')
    elif mode == 'k-means-visualization':
        # Not implemented
        print('Not implmentd yet')

    elif mode == 'bag_of_words':
        assert folder_path is not None, "Need to provide path to images"
        kmeans_filename = os.path.join(save_path, 'kmeans.pkl')
        hcluster_filename = os.path.join(save_path, 'hcluster.pkl')
        assert os.path.exists(kmeans_filename) and os.path.exists(hcluster_filename), "Cannot find kmeans.pkl or hcluster.pkl"
        loaded_kmeans = pickle.load(open(kmeans_filename, 'rb'))
        loaded_hcluster = pickle.load(open(hcluster_filename, 'rb'))
        im_list = glob.glob(os.path.join(folder_path, '*' + ext))
        
    elif mode == 'classifier_train':
        assert save_path is not None and os.path.exists(save_path), "Feature/kmeans path is None or does not exist"
        assert folder_path is not None or image_path is not None, "Error: Input image path is None or save path is None."

        if clf_filename is None:
            # initialize model
            start = True
            clf=model_init(args)
            clf_filename = os.path.join(save_path, 'clf.pkl')
        else:
            clf=model_load(clf_filename)
            start = False
            print('Loaded pre-trained classifier...')

        assert os.path.exists(kmeans) and os.path.exists(hcluster), "Cannot find kmeans.pkl or hcluster.pkl"
        loaded_kmeans = pickle.load(open(kmeans, 'rb'))
        loaded_hcluster = pickle.load(open(hcluster, 'rb'))
        assert loaded_kmeans is not None, "Path incorrect/File doesnt exist"

        if not os.path.exists(os.path.join(save_path, 'trained_file.pkl')):
            processed = []
        else:
            processed = pickle.load(open(os.path.join(save_path,
                                                    'trained_file.pkl'), 'rb'))

        if folder_path is not None:
            pos_dir = os.path.join(folder_path, 'pos')
            pos_files = sorted(glob.glob(os.path.join(pos_dir,
                                                      '*{}'.format(ext))))

            print('Number of positive samples: {}'.format(len(pos_files)))
            neg_dir = os.path.join(folder_path, 'neg')
            neg_files = sorted(glob.glob(os.path.join(neg_dir,
                                                      '*{}'.format(ext))))
            print('Number of negative samples: {}'.format(len(neg_files)))
            i = 0
            while i < len(pos_files) or i < len(neg_files):
                if i % 10 == 0:
                    print("{} / {}".format(i, max(len(pos_files),
                                           len(neg_files))))
                    model_save(clf, clf_filename)
                if i < len(pos_files):
                    im_p = pos_files[i]
                    if (im_p not in processed):
                        print(im_p)
                        bag_feat = get_hist_from_image(im_p, loaded_kmeans,
                                                       loaded_hcluster,
                                                       dict_size,
                                                       word_size)
                        clf = model_update(clf, [bag_feat], [1], start)
                        processed.append(im_p)
                    if start:
                        start = False
                if i < len(neg_files):
                    im_p = neg_files[i]
                    if (im_p not in processed):
                        print(im_p)
                        try:
                            bag_feat = get_hist_from_image(im_p, loaded_kmeans,
                                                           loaded_hcluster,
                                                           dict_size,
                                                           word_size)
                        except np.linalg.LinAlgError:
                            i += 1
                            if start:
                                start = False
                            continue
                        clf = model_update(clf, [bag_feat], [0], start)
                        processed.append(im_p)
                    if start:
                        start = False
                i += 1
                pickle.dump(processed, open(os.path.join(save_path,
                    'trained_file.pkl'), 'wb'))
            model_save(clf, clf_filename)

        elif image_path is not None:
            assert image_label_path is not None and os.path.exists(image_label_path), "Error: invalid label file"

            print('Input training image: {}'.format(image_path))
            image = cv2.imread(image_path)
            image_label = pickle.load(open(image_label_path, 'rb'))
            assert image is not None, "imread fail, check path"
            image = np.array(image, dtype=int)
            bags = Bag(img=image, size=bag_size,
                       overlap_pixel=overlap, padded=True)
            assert len(bags) == len(image_label), "Label and input length does not match"
            for bag, i in bags:
                bag_feat = get_hist_from_image(None, loaded_kmeans,
                           loaded_hcluster, dict_size, word_size,
                           image=bag)
                clf = model_update(clf, [bag_feat], [label[i]], start=False)
                model_save(clf, clf_filename)

    elif mode == 'classifier_test':
        # assert save_path is not None and os.path.exists(save_path), "Feature/kmeans path is None or does not exist"
        assert folder_path is not None or image_path is not None, "Error: Input image path is None or save path is None."
        # assert roi_csv_file is not None, "ROI tracking data not provided"
        # assert wsi_csv_file is not None, "ROI tracking data not provided"
        assert clf_filename is not None, "Error: invalid input model"
        assert os.path.exists(clf_filename), "Error: invalid input model"

        clf=model_load(clf_filename)
        print('Loaded pre-trained classifier...')

        assert os.path.exists(kmeans) and os.path.exists(hcluster), "Cannot find kmeans.pkl or hcluster.pkl"
        loaded_kmeans = pickle.load(open(kmeans, 'rb'))
        loaded_hcluster = pickle.load(open(hcluster, 'rb'))
        assert loaded_kmeans is not None, "Path incorrect/File doesnt exist"
        assert loaded_hcluster is not None, "Path incorrect/File doesnt exist"

        print('Loaded kmeans and hclusters...')

        # dict_bbox = preprocess_roi_csv(roi_csv)
        # dict_wsi_size = preprocess_wsi_size_csv(wsi_size_csv)

        if folder_path is not None:
            caseIDs = get_immediate_subdirectories(folder_path)
            for caseID in caseIDs:
                if any(char.isdigit() for char in caseID):
                    print('-------Processing: {}-------'.format(caseID))
                    caseID = int(caseID)
                    im_list = glob.glob(os.path.join(folder_path, 
                                                            str(caseID), '*' +
                                                            ext))
                    metrics_list = [{'accuracy': 0, 'metrics':(0, 0, 0, 0)}]*len(im_list)
                    index = 0
                    filename =  'test_result.pkl'
                    out_p = os.path.join(folder_path, str(caseID), filename)
                    if os.path.exists(out_p):
                        continue
                    for im_p in im_list:
                        base, file_extend = os.path.splitext(im_p)
                        if 'openslide' in sys.modules:
                            im = openslide.OpenSlide(im_p)
                            im_size = (im.dimensions[1], im.dimensions[0])
                            bags = Bag(h=im_size[0],
                                       w=im_size[1], size=bag_size,
                                       overlap_pixel=overlap,
                                       padded=True)
                            result = np.zeros(len(bags))
                            for i in range(len(bags)):
                                bbox = bags.bound_box(i)
                                size_r = bbox[1] - bbox[0]
                                size_c = bbox[3] - bbox[2]
                                top_left = (bbox[2], bbox[0])
                                bag = im.read_region(top_left, image_level,
                                                     (size_c,
                                                      size_r)).convert('RGB')
                                bag = np.array(bag, dtype=np.uint8)
                                # bag = cv2.cvtColor(bag, cv2.COLOR_RGB2BGR)
                                if check_empty(bag):
                                    result[i] = 0
                                    continue
                                bag_feat = get_hist_from_image(None,
                                                               loaded_kmeans,
                                                               loaded_hcluster,
                                                               dict_size,
                                                               word_size,
                                                               image=bag)
                                result[i] = model_predict(clf, [bag_feat])

                        else:
                            l_p = base + '_label.pkl'
                            assert os.path.exists(l_p), "Did not find corresponding label file for {}".format(caseID)
                            print('-------Processing: {}-------'.format(os.path.basename(base)))
                            result_outpath =  '{}_result.pkl'.format(base)
                            feat_output = '{}_feat.pkl'.format(base)
                            image_label = pickle.load(open(l_p, 'rb'))
                            if not os.path.exists(result_outpath):
                                image = cv2.imread(im_p)
                                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                                assert image is not None, "imread fail, check path"
                                image = np.array(image, dtype=int)
                                bags = Bag(h=image.shape[0],
                                               w=image.shape[1], size=bag_size,
                                               overlap_pixel=overlap,
                                               padded=True)
                                feat = np.zeros([len(bags), dict_size]) - 1
                                try:
                                    bags_feat = get_hist_from_large_image(None, loaded_kmeans,loaded_hcluster, bag_size,
                                        dict_size, word_size,
                                        histogram_bin=histogram_bin,
                                        overlap_pixel=overlap,
                                        image=image)
                                    feat[i, :] = bag_feat
                                except np.linalg.LinAlgError:
                                    result[i] = 0
                                    all_blank = [0] * 40
                                    all_blank[24] = 900
                                    feat[i] = all_blank

                                result = model_predict(clf, bags_feat)
                                pickle.dump(result, open(result_outpath, 'wb'))
                                pickle.dump(feat, open(feat_outpath, 'wb'))
                                accuracy, metrics = model_report(result,
                                                                 image_label,
                                                                 train=False)

                                '''
                                Old Implementation using get_hist_from_image
                                bags = Bag(img=image, size=bag_size,
                                           overlap_pixel=overlap, padded=True)
                                assert len(bags) == len(image_label), "Label and input length does not match"
                                result = np.zeros(len(bags))
                                
                                for bag, i in bags:
                                    bag_feat = get_hist_from_image(None, loaded_kmeans,
                                               loaded_hcluster, dict_size, word_size,
                                               image=bag)
                                    result[i] = model_predict(clf, [bag_feat])
                                '''
                            else:
                                result = pickle.load(open(result_outpath,
                                                          'rb'))
                            
                            accuracy, metrics = model_report(result,
                                                             image_label,
                                                             train=False)

                            metrics_list[index]['accuracy'] = accuracy
                            metrics_list[index]['metrics'] = metrics
                            index += 1
                    pickle.dump(metrics_list, open(out_p, 'wb'))
        else:
            assert image_path is not None

            if save_path is not None:
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                base, _ = os.path.splitext(os.path.basename(image_path))
                result_outpath = os.path.join(save_path,
                                              '{}_result.pkl'.format(base))
                feat_outpath = os.path.join(save_path,
                                            '{}_feat.pkl'.format(base))
            else:
                base, file_extend = os.path.splitext(image_path)
                result_outpath = '{}_result.pkl'.format(base)
                feat_outpath = '{}_feat.pkl'.format(base)

            im = openslide.OpenSlide(image_path)
            im_size = (im.dimensions[1], im.dimensions[0])
            bags = Bag(h=im_size[0],
                       w=im_size[1], size=bag_size,
                       overlap_pixel=overlap,
                       padded=True)
            result = np.zeros(len(bags)) - 1
            feat = np.zeros([len(bags), dict_size]) - 1
            for i in range(len(bags)):
                if i % 50 == 0:
                    print(result)
                    print("{} / {}".format(i, len(bags)))
                bbox = bags.bound_box(i)
                size_r = bbox[1] - bbox[0]
                size_c = bbox[3] - bbox[2]
                top_left = (bbox[2], bbox[0])
                bag = im.read_region(top_left, image_level,
                                     (size_c,
                                      size_r)).convert('RGB')
                bag = np.array(bag, dtype=np.uint8)
                # bag = cv2.cvtColor(bag, cv2.COLOR_RGB2BGR)
                try:
                    bag_feat = get_hist_from_image(None,
                                                   loaded_kmeans,
                                                   loaded_hcluster,
                                                   dict_size,
                                                   word_size,
                                                   image=bag)
                    feat[i, :] = bag_feat
                    result[i] = model_predict(clf, [bag_feat])
                    # print(bag_feat)
                    # print(result[i])
                except np.linalg.LinAlgError:
                    result[i] = 0
                    all_blank = [0] * 40
                    all_blank[23] = 900
                    feat[i] = all_blank

                pickle.dump(result, open(result_outpath, 'wb'))

            assert -1 not in result
            assert -1 not in feat

            if image_label_path is not None:
                image_label = pickle.load(open(image_label_path, 'rb'))
                accuracy, metrics = model_report(result,
                                                 image_label,
                                                 train=False)

            pickle.dump(feat, open(feat_outpath, 'wb'))
