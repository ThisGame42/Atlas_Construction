import numpy as np
import nibabel as nib
import torch
import torchio as tio
import torch.nn.functional as F

from torch.utils.data import Dataset
from Utils.utils import read_n_permute, load_from_f


def get_aug_data(ident, original_files, num_iter):
    aug_mdixon_files = list()
    for mdixon_file in original_files:
        for i in range(num_iter):
            aug_mdixon_file = mdixon_file.replace(ident, f"{ident}_{i}")
            aug_mdixon_files.append(aug_mdixon_file)
    aug_mdixon_files = sorted(original_files + aug_mdixon_files)
    return aug_mdixon_files


class Dataset2D(Dataset):
    def __init__(self,
                 template_file,
                 template_file_label,
                 img_path,
                 label_path,
                 num_classes=12,
                 for_alignment=False,
                 batch_size=128):
        mdixon_files = load_from_f(img_path)
        mdixon_labels = load_from_f(label_path)
        self.mdixon_files = mdixon_files
        self.num_classes = num_classes
        self.batch_size = batch_size

        self.img_count, self.label_count = 0, 0
        assert len(mdixon_files) == len(mdixon_labels)
        self.rescale = tio.RescaleIntensity(out_min_max=(-1, 1))

        self.data = list()
        self.num_sub_volumes = 512
        self.sub_img, self.sub_label = list(), list()
        self.whole_img, self.whole_label = list(), list()
        template_data = tio.Subject(image=tio.ScalarImage(tensor=read_n_permute(template_file)))
        template_data = self.rescale(template_data)
        self.template = template_data["image"].data.clone()
        self.template_affine = nib.load(template_file).affine
        template_label = tio.Subject(label=tio.ScalarImage(tensor=read_n_permute(template_file_label)))
        self.template_label = template_label["label"].data.permute(0, 3, 1, 2).clone()
        self.template_label_affine = nib.load(template_file_label).affine
        for idx, (mdixon, m_labels) in enumerate(zip(mdixon_files, mdixon_labels)):
            if for_alignment:
                if "_r.nii.gz" not in mdixon:
                    continue
            mdixon_data = tio.Subject(image=tio.ScalarImage(tensor=read_n_permute(mdixon)),
                                      label=tio.LabelMap(tensor=read_n_permute(m_labels)))
            mdixon_data = self.rescale(mdixon_data)
            num_slices = mdixon_data["image"].data.shape[-1]
            for i in range(num_slices):
                m_data = torch.squeeze(mdixon_data["image"].data[..., i], dim=0)
                m_label = mdixon_data["label"].data[..., i]
                self.data.append([i,
                                  m_data,
                                  m_label,
                                  mdixon])

    def get_sub_template(self, i):
        sub_template = torch.squeeze(self.template[..., i], dim=0)
        sub_template_label = self.template_label[:, i, ...]
        return sub_template, sub_template_label

    def __len__(self):
        return len(self.data)

    def get_num_files(self):
        return self.num_files

    def get_files(self):
        return self.mdixon_files

    def __getitem__(self, idx):
        idx, fixed_img, fixed_labels, mdixon_name = self.data[idx]
        moving_atlas, moving_atlas_label = self.get_sub_template(idx)
        binary_labels = fixed_labels.clone()
        binary_labels[binary_labels != 0] = 1
        fixed_labels = F.one_hot(torch.squeeze(fixed_labels.to(torch.int64), dim=0),
                                 num_classes=12)
        fixed_labels = fixed_labels.permute(2, 0, 1)
        # already one hot
        return {
            "moving_atlas": moving_atlas.to(torch.float32),
            "fixed_img": fixed_img.to(torch.float32),
            "moving_atlas_label": moving_atlas_label.to(torch.float32),
            "fixed_labels": fixed_labels.to(torch.float32),
            "mdixon_name": mdixon_name,
            "binary_labels": binary_labels.to(torch.float32),
            "index": idx
        }

    def insert_warped_img(self, sub_img):
        sub_img = sub_img.permute(1, 0, 2, 3)
        self.sub_img.append(sub_img.detach().cpu().numpy())
        self.img_count += 1
        if self.img_count == (512 // self.batch_size):
            whole_img = np.zeros((4, 512, 240, 240))
            for idx, i in enumerate(range(0, self.num_sub_volumes, self.batch_size)):
                start_idx = i
                end_idx = i + self.batch_size
                whole_img[:, start_idx:end_idx] = self.sub_img[idx]
            self.whole_img.append(whole_img)
            self.sub_img.clear()
            self.img_count = 0

    def insert_warped_label(self, sub_label):
        sub_label = sub_label.permute(1, 0, 2, 3)
        self.sub_label.append(sub_label.detach().cpu().numpy())
        self.label_count += 1
        if self.label_count == 512 // self.batch_size:
            whole_label = np.zeros((12, 512, 240, 240))
            for idx, i in enumerate(range(0, self.num_sub_volumes, self.batch_size)):
                start_idx = i
                end_idx = i + self.batch_size
                whole_label[:, start_idx:end_idx, ...] = self.sub_label[idx]
            whole_label = torch.from_numpy(whole_label)
            self.whole_label.append(whole_label.numpy())
            self.sub_label.clear()
            self.label_count = 0

    def update_template_img(self, alpha):
        average_template_new = np.mean(np.array(self.whole_img), axis=0)
        average_template_new = torch.from_numpy(average_template_new).permute(0, 2, 3, 1)
        self.template = (1 - alpha) * self.template + alpha * average_template_new

        # rescale intensity
        self.template = self.rescale(tio.Subject(
            image=tio.ScalarImage(tensor=self.template)
        ))["image"].data
        self.whole_img.clear()

    def update_template_label(self, alpha):
        template_label_new = torch.from_numpy(np.mean(np.array(self.whole_label), axis=0))
        self.template_label = (1 - alpha) * self.template_label + alpha * template_label_new
        self.whole_label.clear()

    def retrieve_template(self):
        return self.template, self.template_affine

    def retrieve_template_label(self):
        return self.template_label, self.template_label_affine
