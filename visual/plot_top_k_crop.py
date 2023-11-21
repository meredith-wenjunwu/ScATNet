import glob
import tqdm
import os
import math
import argparse
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000
import numpy as np
import cv2

def read_original_image(path, crop_size1, crop_size2, mask=None, output_mask=None):
    image = Image.open(path)
    w, h = image.size
    if h > w:
        image = image.rotate(90, expand=True)
    w, h = image.size
    assert w >= h

    if mask is not None:
        bn = os.path.basename(path)
        im_ind = os.path.splitext(bn)[0]
        stage = os.path.basename(os.path.dirname(os.path.dirname(path)))
        mask_path = os.path.join(mask, stage,
                                 im_ind.split('_z0')[0].replace('S2_', '') + '_z0', '{}.png'.format(im_ind))
        mask = Image.open(mask_path).convert('L')
        w_m, h_m = mask.size
        if h_m > w_m:
            mask = mask.rotate(90, expand=True)
        # apply mask
        mask = np.uint8(mask)
        # background --> value 0 in mask
        # nonbackground = np.uint8(mask == 0) * 255
        # nonbackground = Image.fromarray(nonbackground)
        # image.paste((0, 0, 0), mask=nonbackground)

        # apply mask and warped image accordingly
        image, mask = warped_rotated_crops(np.array(image, dtype=np.uint8), mask)
        image = Image.fromarray(image)
        w, h = image.size
        if h > w:
            image = image.rotate(90, expand=True)
        w, h = image.size
        assert w >= h
        mask = Image.fromarray(mask)
        w_m, h_m = mask.size
        if h_m > w_m:
            mask = mask.rotate(90, expand=True)
        mask.save(output_mask)
        #cv2.imwrite(output_mask, mask)

    if crop_size1 is not None and len(crop_size1) > 0:
        image = image.resize((crop_size1[1], crop_size1[0]), Image.BICUBIC)
    else:
        new_w = math.ceil(w / crop_size2) * crop_size2
        new_h = math.ceil(h / crop_size2) * crop_size2
        image = image.resize((new_w, new_h), Image.BICUBIC)
    return image


def read_file_ind_from_txt(file_path):
    assert os.path.exists(file_path)
    with open(file_path, 'r') as f:
        lines = [line.rstrip() for line in f]
    image_path = lines[0]
    assert os.path.exists(image_path)
    indices = [int(x) for x in lines[1:]]
    return image_path, indices


def calculate_bbox(img_size, resize2, idx):
    """
    Function that return the bounding box of a word given its index
    Args:
        ind: int, ind < number of words

    Returns:
        Bounding box(int[]): [h_low, h_high, w_low, w_high]
    """
    h, w = img_size
    c_w = w // resize2
    c_h = h // resize2
    crop_length = c_w * c_h
    assert idx < crop_length, "Index Out of Bound"

    # [index]: [pad_top, pad_left, pad_right, pad_bottom]
    # top= max((idx % c_h) * (resize2), 0)
    top = max(math.floor(idx / c_w) * resize2, 0)
    # bottom = min(h, (idx % c_h) * resize2+ resize2)
    bottom = min(h, math.floor(idx / c_w) * resize2 + resize2)
    left = max((idx % c_w) * resize2, 0)
    right = min(w, (idx % c_w) * resize2 + resize2)
    # left = max(math.floor(idx / c_h) * resize2, 0)
    # right = min(w, math.floor(idx / c_h) * resize2 + resize2)

    return [top, bottom, left, right]


def draw_rectangle_on_bbox(image, bbox, thickness):
    output = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    output[bbox[0]:bbox[1], bbox[2]:bbox[3]] = 255
    contours, hierarchy = cv2.findContours(output,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    image = cv2.drawContours(image, contours, -1, (0, 0, 255), thickness)
    return image


def warped_rotated_crops(image, mask):
    contours, hierarchy = cv2.findContours(mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    cnt = np.vstack([x for x in contours])
    M = get_rotation_matrix_cnts(cnt)
    rect = cv2.minAreaRect(cnt)
    width = int(rect[1][0])
    height = int(rect[1][1])
    warped_crop = cv2.warpPerspective(image, M, (width, height),
                                      borderValue=(0, 0, 0))
    warped_mask = cv2.warpPerspective(mask, M, (width, height), borderValue=0)
    warped_mask = warped_mask > 200
    warped_mask = warped_mask * 255
    warped_mask = np.array(warped_mask, dtype=np.uint8)
    return warped_crop, warped_mask


def get_rotation_matrix_cnts(cnt):
    rect = cv2.minAreaRect(cnt)
    # the order of the box points: bottom left, top left,
    # top right, bottom right
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    # get width and height of the detected rectangle
    width = int(rect[1][0])
    height = int(rect[1][1])

    src_pts = box.astype("float32")
    # corrdinate of the points in box
    # points after the rectangle has been
    # straightened
    dst_pts = np.array([[0, height - 1],
                        [0, 0],
                        [width - 1, 0],
                        [width - 1, height - 1]],
                       dtype="float32")
    # the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    return M


def main():
    parser = argparse.ArgumentParser(description='Plot top k crops in whole slide images')
    parser.add_argument('--results-dir', required=True, type=str, default='results', help='results directory location.')
    parser.add_argument('--top-k', required=True, type=int, default=10, help='Num of crops to plot')
    parser.add_argument('--resize1', '--crop1', type=int, nargs='+')
    parser.add_argument('--resize2', '--crop2', default=256, type=int)
    parser.add_argument('--plot-thickness', type=int, default=20)
    parser.add_argument('--mask', type=str, help="path to mask folder if used")
    args = parser.parse_args()
    results_dir = args.results_dir
    for label_dir in os.listdir(results_dir):
        label_dir = os.path.join(results_dir, label_dir)
        if not os.path.isdir(label_dir):
            continue
        print('----processing: {} ----'.format(os.path.basename(label_dir)))
        for case_dir in tqdm.tqdm(os.listdir(label_dir)):
            case_dir = os.path.join(label_dir, case_dir)
            if not os.path.isdir(case_dir):
                continue
            output_path = os.path.join(case_dir, 'top_{}_marked.jpg'.format(args.top_k))
            # if os.path.exists(output_path):
            #     print(output_path)
            #     continue
            img_path, top_k_indices = read_file_ind_from_txt(os.path.join(case_dir, 'indices.txt'))
            top_k_indices = top_k_indices[:args.top_k]
            output_mask = os.path.join(case_dir, 'mask.jpg')
            image = read_original_image(img_path, args.resize1, args.resize2, mask=args.mask, output_mask=output_mask)
            image = np.array(image, dtype=np.uint8)
            im_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            for ind in top_k_indices:
                # pdb.set_trace()
                bbox = calculate_bbox(args.resize1, args.resize2, ind)
                im_BGR = draw_rectangle_on_bbox(im_BGR, bbox, args.plot_thickness)
            cv2.imwrite(output_path, im_BGR)


if __name__ == '__main__':
    import pdb
    main()

