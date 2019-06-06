import argparse
import base64
import json
import os
import os.path as osp
import warnings

import PIL.Image
import yaml

from labelme import utils
###############################################增加的语句,改下路径即可##############################
import glob


###############################################   end    ##################################


def json2label(json_folder):
    json_list = glob.glob(os.path.join(json_folder, '*.json'))

    annotations_dir = os.path.join(json_folder, "annotations")
    josn2labels_dir = os.path.join(json_folder, "josn2labels")

    if not os.path.exists(annotations_dir):
        os.makedirs(annotations_dir)
    if not os.path.exists(josn2labels_dir):
        os.makedirs(josn2labels_dir)

    for json_file in json_list:
        temp_name = json_file.split(".")[0].split("/")[-1]

        out_dir = os.path.join(josn2labels_dir, temp_name)
        if not osp.exists(out_dir):
            os.mkdir(out_dir)

        data = json.load(open(json_file))

        if data['imageData']:
            imageData = data['imageData']
        else:
            imagePath = os.path.join(os.path.dirname(json_file), data['imagePath'])
            with open(imagePath, 'rb') as f:
                imageData = f.read()
                imageData = base64.b64encode(imageData).decode('utf-8')
        img = utils.img_b64_to_arr(imageData)

        label_name_to_value = {'_background_': 0, '1': 1, '2': 2, '3': 3}
        # for shape in sorted(data['shapes'], key=lambda x: x['label']):
        #     label_name = shape['label']
        #     if label_name in label_name_to_value:
        #         label_value = label_name_to_value[label_name]
        #     else:
        #         label_value = len(label_name_to_value)
        #         label_name_to_value[label_name] = label_value
        lbl = utils.shapes_to_label(img.shape, data['shapes'], label_name_to_value)

        label_names = [None] * (max(label_name_to_value.values()) + 1)
        for name, value in label_name_to_value.items():
            label_names[value] = name
        lbl_viz = utils.draw_label(lbl, img, label_names)

        PIL.Image.fromarray(img).save(osp.join(out_dir, '%s_img.png' % (temp_name)))
        utils.lblsave(osp.join(out_dir, '%s.png' % (temp_name)), lbl)
        utils.lblsave(osp.join(annotations_dir, '%s.png' % (temp_name)), lbl)
        PIL.Image.fromarray(lbl_viz).save(osp.join(out_dir, '%s_label_viz.png' % (temp_name)))

        with open(osp.join(out_dir, 'label_names.txt'), 'w') as f:
            for lbl_name in label_names:
                f.write(lbl_name + '\n')

        warnings.warn('info.yaml is being replaced by label_names.txt')
        info = dict(label_names=label_names)
        with open(osp.join(out_dir, 'info.yaml'), 'w') as f:
            yaml.safe_dump(info, f, default_flow_style=False)

        print('Saved to: %s' % out_dir)


if __name__ == '__main__':
    json_folder = '/home/alpha/Desktop/images/json'

    json2label(json_folder)
