import os
import glob
import shutil
import subprocess


def divert_data(data_path, image_path, label_path):
    files = sorted(glob.glob(os.path.join(data_path, "*.nii.gz")))
    images = [f for f in files if "mdixon.nii.gz" in f]
    labels = [f for f in files if "mdixon_segmentation.nii.gz" in f]

    for f in images:
        base_name = os.path.basename(f)
        shutil.move(f, os.path.join(image_path, base_name))

    for f in labels:
        base_name = os.path.basename(f)
        base_name = base_name.replace("_segmentation", "")
        shutil.move(f, os.path.join(label_path, base_name))


def produce_template(img_path, template_path, output_path, linear_path, warp_dir):
    cmd = ["population_template", img_path, template_path,
           "-voxel_size", "1,1,1", "-initial_alignment", "mass", "-leave_one_out", "1",
           "-force", "-linear_transformations_dir", linear_path, "-transformed_dir", output_path,
           "-type", "rigid_affine", "-nthreads", "16", "-linear_no_pause"] #"-warp_dir", warp_dir]
    subprocess.run(cmd)


if __name__ == "__main__":
    produce_template("/media/jiayi/Data/Dataset/dixon_template/dixon_img",
                     "/media/jiayi/Data/Dataset/dixon_template/template_img.nii.gz",
                     "/media/jiayi/Data/Dataset/dixon_template/transformed",
                     "/media/jiayi/Data/Dataset/dixon_template/linear_trans",
                     "/media/jiayi/Data/Dataset/dixon_template/linear_warp")
