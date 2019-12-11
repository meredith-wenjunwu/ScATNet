from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
from tkinter.ttk import Progressbar
from PIL import ImageTk, Image
from bag import Bag
from util import get_feat_from_image, get_histogram_cluster, biggest_bbox
from classifier import *
from cluster import predict_kmeans
import numpy as np
import os
os.environ['OPENCV_IO_MAX_IMAGE_PIXELS']=str(2**64)
import cv2

root = Tk()
root.title('MLCD')

model_path = StringVar()
model_path.set('Model Path Goes Here')
input_path = StringVar()
input_path.set('Input Image Path Goes Here')
output_path = StringVar()
output_path.set('Output Path Goes Here')
run_flag = BooleanVar()

progressbar = Progressbar(root, orient=HORIZONTAL,
                          length=250,mode='determinate')
progressbar.place(anchor="w")


def get_input():
    global input_path, output_path
    filename = filedialog.askopenfilename(initialdir="./",
                                          title="Select image file",
                                          filetypes=(("jpeg files", "*.jpg"),
                                                     ("tif flile", "*.tif"),
                                                     ("tiff file", "*.tiff"),
                                                     ("jpeg file", "*.jpg"),
                                                     ("png file", "*.png"),
                                                     ("svs file", "*.svs")))

    dirname = os.path.dirname(filename)
    input_path.set(filename)
    if output_path == 'Saved Image Path Goes Here':
        output_put.set(dirname)


def load_model():
    global model_path
    foldername = filedialog.askdirectory(title='Path to downloaded Models')
    model_path.set(foldername)
    #print(model_path.get())

def begin_task():
    root.update()
    model_p = model_path.get()
    if model_p == 'Model Path Goes Here':
        load_model()
    input_p = input_path.get()
    if input_p == 'Input Image Path Goes Here':
        get_input()
    output_p = output_path.get()
    clf_filename = os.path.join(model_p, 'clf.pkl')
    kmeans_filename = os.path.join(model_p, 'kmeans.pkl')
    hcluster_filename = os.path.join(model_p, 'hcluster.pkl')
    if not os.path.exists(clf_filename):
        clf_filename = filedialog.askopenfilename(initialdir="./",
        title="Select Trained SVM Model File",
        filetypes=(("Pickle File", "*.pkl"),
                   ("all files", "*.*")))
    clf=model_load(clf_filename)
    if not os.path.exists(kmeans_filename):
        kmeans_filename = filedialog.askopenfilename(initialdir="./",
        title="Select Trained K-Means Model File",
        filetypes=(("Pickle File", "*.pkl"),
                   ("all files", "*.*")))
    if not os.path.exists(hcluster_filename):
        hcluster_filename = filedialog.askopenfilename(initialdir="./",
        title="Select Trained H-cluster File",
        filetypes=(("Pickle File", "*.pkl"),
                   ("all files", "*.*")))
    loaded_kmeans = pickle.load(open(kmeans_filename, 'rb'))
    loaded_hcluster = pickle.load(open(hcluster_filename, 'rb'))
    progressbar['value'] = 0
    percent['text'] = "{}%".format(progressbar['value'])
    root.update()
    im_BGR = cv2.imread(input_p)
    if im_BGR is None:
        messagebox.showerror("Error", "CV2 image read error: image must " +
                                      "have less than 2^64 pixels")

    im = cv2.cvtColor(im_BGR, cv2.COLOR_BGR2RGB)
    im = np.array(im, dtype=np.uint8)
    output = np.empty((im.shape[0], im.shape[1]))
    bags = Bag(img=im, size=3600,
               overlap_pixel=2400, padded=True)
    bn = os.path.basename(input_p)
    bn = os.path.splitext(bn)[0]
    feat_outname = os.path.join(output_p, '{}_feat.pkl'.format(bn))
    if os.path.exists(feat_outname):
        feat = pickle.load(open(feat_outname, 'rb'))
        precomputed = True
    else:
        feat = np.zeros([len(bags), 40])
        precomputed = False

    result = np.zeros(len(bags))
    # base = 20
    for bag, i in bags:
        # print('{}/{}'.format(i, len(bags)))
        # cv2.imwrite(os.path.join(output_p, '{}.jpg'. format(i)),
        #             cv2.cvtColor(bag, cv2.COLOR_RGB2BGR))
        # if (float(i) / len(bags)) * 100 > base:
        progressbar['value'] = min((float(i+1) / len(bags)) * 100, 100)
        percent['text'] = "{:.1f}%".format(progressbar['value'])
        root.update()
        # base = min(100, base + 10)
        if not precomputed:
            try:
                feat_words = get_feat_from_image(None, False, 120, image=bag)
                cluster_words = predict_kmeans(feat_words, loaded_kmeans,
                                               h_cluster=loaded_hcluster)
                hist_bag = get_histogram_cluster(cluster_words,
                                                 dict_size=40)
            except np.linalg.LinAlgError:
                result[i] = 0
                hist_bag = [0] * 40
                hist_bag[23] = 900
            feat[i, :] = hist_bag
            pickle.dump(feat, open(feat_outname, 'wb'))
        result[i] = model_predict(clf, [feat[i, :]])
        # print('result: {}'.format(result[i]))
        bbox = bags.bound_box(i)
        bbox[0] = max(0, min(bbox[0] - bags.top, im.shape[0] - 1))
        bbox[1] = max(0, min(bbox[1] - bags.top, im.shape[0] - 1))
        bbox[2] = max(0, min(bbox[2] - bags.left, im.shape[1] - 1))
        bbox[3] = max(0, min(bbox[3] - bags.left, im.shape[1] - 1))
        output[bbox[0]:bbox[1], bbox[2]:bbox[3]] = result[i]
        # if result[i] == 1:
        #     cv2.imwrite(os.path.join(output_p, '{}_binary.jpg'. format(i)),
        #                 np.array(output * 255, dtype=np.uint8))
        #     cv2.imwrite(os.path.join(output_p, '{}.jpg'. format(i)),
        #                 im_BGR[bbox[0]:bbox[1], bbox[2]:bbox[3]])

    # draw bounding box and save
    output *= 255
    output = np.array(output, dtype=np.uint8)
    # save image
    pickle.dump(feat, open(feat_outname, 'wb'))
    #binary_outname = os.path.join(output_p, '{}_binary.jpg'.format(bn))
    #cv2.imwrite(binary_outname, output)
    contours, hierarchy = cv2.findContours(output,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    final = im_BGR.copy()
    final = cv2.drawContours(final, contours, -1, (0, 0, 255), 8)
    marked_outname = os.path.join(output_p, '{}_marked.jpg'.format(bn))
    cv2.imwrite(marked_outname, final)

    # save jpeg for segmentation
    count = 0
    bboxes = []
    for cont in contours:
        x, y, w, h = cv2.boundingRect(cont)
        img = im_BGR[y:y + h, x:x + w, :]
        bboxes += [[y, y + h, x, x + w]]
        roi_outname = os.path.join(output_p, '{}_{}.jpg'.format(bn, count))
        cv2.imwrite(roi_outname, img)
    # print(bboxes)

    # # scale result and display
    # box = biggest_bbox(bboxes)
    # box[0] = max(box[0] - 20, 0)
    # box[1] = min(box[1] + 20, final.shape[0])
    # box[2] = max(box[2] - 20, 0)
    # box[3] = min(box[3] + 20, final.shape[1])
    # w = box[3] - box[2]
    # h = box[1] - box[0]

    draw_area = final.copy()

    scale_side = max(draw_area.shape[0], draw_area.shape[1])
    if scale_side > 800:
        scale_factor = float(scale_side) / 500
        final_resized = cv2.resize(draw_area, None, fx=1 / scale_factor,
                                   fy=1 / scale_factor,
                                   interpolation=cv2.INTER_AREA)
    else:
        final_resized = draw_area
    #resized_outname = os.path.join(output_p, '{}_vis.jpg'. format(bn))
    #cv2.imwrite(resized_outname, final_resized)
    display_im2 = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(final_resized,
                                                                  cv2.COLOR_BGR2RGB)))
    im_label.configure(image=display_im2)
    im_label.image = display_im2
    root.update()


def get_outdir():
    global output_path
    foldername = filedialog.askdirectory(title='Select Output Directory')
    output_path.set(foldername)


title = Label(root,
              text='ROIWindowClassifier',
              font=('Arial', 24),
              width=50, height=1)
title.grid(row=0, column=0, columnspan=3)

button_input = Button(root, text="Select Input Image",
                      command=get_input)
button_predict = Button(root, text="Predict",
                        command=begin_task,
                        width=40, height=2)
button_trained_model = Button(root, text="Select Pre-trained Model Path",
                              command=lambda: load_model())
button_output = Button(root, text="Select Output Path",
                       command=get_outdir)

display_im = ImageTk.PhotoImage(Image.open('./test.jpg'))
im_label = Label(root, image=display_im)
percent = Label(root, text="", justify=LEFT)
model_path_label = Label(root, textvariable=model_path)
progress_label = Label(root, text="Progress: ")
outpath_label = Label(root, textvariable=output_path)
inpath_label = Label(root, textvariable=input_path)


im_label.grid(row=7, column=0, columnspan=3, pady=15)

button_input.grid(row=2, column=0)
button_trained_model.grid(row=1, column=0)
button_output.grid(row=3, column=0)
button_predict.grid(row=6, column=0, columnspan=3, pady=(5,0))

model_path_label.grid(row=1, column=1)
progress_label.grid(row=4, column=0)
progressbar.grid(row=4, column=1, pady=(5, 0))
percent.grid(row=4, column=2, sticky='W')
outpath_label.grid(row=3, column=1)
inpath_label.grid(row=2, column=1)
root.mainloop()
