import torch
import numpy as np
import nibabel as nib


class Tester(object):
    def __init__(self, netG, output_path, loader_t, dataset_t, device):
        self.output_path = output_path
        self.netG = netG
        self.loader_t = loader_t
        self.dataset_t = dataset_t
        self.device = device

        self.netG.eval()

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

    def test_run(self):
        with torch.no_grad():
            for batch_data in self.loader_t:
                results = self.test_generator_one_step(batch_data)
                r1 = self.dataset_t.insert_test_img(results["moved_img"])
                r2 = self.dataset_t.insert_test_label(results["moved_label"])
                if type(r1) is not bool:
                    moved_img, moved_label = r1, r2
                    # get image name, label name, calculate metrics, print them out
