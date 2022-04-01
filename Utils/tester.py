import os
import json
import torch
import nibabel as nib
import torch.nn.functional as F

from Utils.eval import eval


class Tester(object):
    def __init__(self, netG, output_path, loader_t, dataset_t, device):
        self.output_path = output_path
        self.netG = netG
        self.loader_t = loader_t
        self.dataset_t = dataset_t
        self.device = device

        self.netG.eval()

        self.img_results = dict()
        self.label_results = dict()

    def generate(self, batch_data):
        moving_atlas, fixed_img, moving_atlas_label, fixed_labels = batch_data["moving_atlas"].to(self.device), \
                                                                    batch_data["fixed_img"].to(self.device), \
                                                                    batch_data["moving_atlas_label"].to(self.device), \
                                                                    batch_data["fixed_labels"].to(self.device)
        results = self.netG(moving_img=moving_atlas,
                            fixed_img=fixed_img,
                            moving_label=moving_atlas_label,
                            fixed_label=fixed_labels)
        return results

    def test_generator_one_step(self, batch_data):
        return self.generate(batch_data)


    def get_test_pairs(self, batch_data):
        img_name, label_name = batch_data["mdixon_name"][0], batch_data["label_name"][0]
        img = torch.from_numpy(nib.load(img_name).get_fdata()).permute(3, 2, 0, 1)
        label = F.one_hot(torch.from_numpy(nib.load(label_name).get_fdata()).permute(2, 0, 1).to(torch.int64),
                          num_classes=12).permute(3, 0, 1, 2)
        return img, label, img_name, label_name



    def test_run(self):
        with torch.no_grad():
            for batch_data in self.loader_t:
                results = self.test_generator_one_step(batch_data)
                r1 = self.dataset_t.insert_test_img(results["moved_img"])
                r2 = self.dataset_t.insert_test_label(results["moved_label"])
                if type(r1) is not bool:
                    moved_img, moved_label = r1, r2
                    fixed_img, fixed_label, img_name, label_name = self.get_test_pairs(batch_data)
                    img_results, label_results = eval(pred_img=moved_img, ref_img=fixed_img,
                                                      pred_label=moved_label, ref_label=fixed_label)
                    assert img_name not in self.img_results.keys()
                    assert label_name not in self.label_results.keys()
                    self.img_results[img_name] = img_results
                    self.label_results[label_results] = label_results

    def save_results(self):
        with open(os.path.join(self.output_path, "img_results.json"), "w+") as f:
            json.dump(self.img_results, f)

        with open(os.path.join(self.output_path, "label_results.json"), "w+") as f:
            json.dump(self.label_results, f)
