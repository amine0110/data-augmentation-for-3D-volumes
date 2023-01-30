[![GitHub issues](https://img.shields.io/github/issues/amine0110/data-augmentation-for-3D-volumes)](https://github.com/amine0110/data-augmentation-for-3D-volumes/issues) [![GitHub license](https://img.shields.io/github/license/amine0110/data-augmentation-for-3D-volumes)](https://github.com/amine0110/data-augmentation-for-3D-volumes) [![GitHub stars](https://img.shields.io/github/stars/amine0110/data-augmentation-for-3D-volumes)](https://github.com/amine0110/data-augmentation-for-3D-volumes/stargazers) ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/py) [![YouTube Video Views](https://img.shields.io/youtube/views/bh9uyUbsj7U?style=social)](https://youtu.be/bh9uyUbsj7U) ![GitHub watchers](https://img.shields.io/github/watchers/amine0110/data-augmentation-for-3D-volumes?style=social)
# 3D volumes augmentation for medical images

## Introduction
We discussed how to preprocess 3D volumes for tumor segmentation in the [previous article](https://pycad.co/preprocessing-3d-volumes-for-tumor-segmentation-using-monai-and-pytorch/), so in this article we will discuss another important step when working on a deep learning project. This is the data augmentation step.

## What is Data Augmentation?
We are all aware that in order to train a neural network, a significant amount of data is required in order to obtain an accurate model as well as a robust model that can work with the majority of cases in that specific task. However, it is not always possible to obtain a large amount of natural data in any task, particularly in healthcare projects. Because one input in medical imaging is a single patient with multiple slices, and we all know how difficult it is to assemble a dataset of this type of data (a lot of patients).

For this project we are gonna need `Python`, `PyTorch` and `Monai`.


----------------------------------------------------------

## Little explanation

```Python
train_images = sorted(glob.glob(os.path.join(data_dir, "TrainData", "*.nii.gz")))
train_labels = sorted(glob.glob(os.path.join(data_dir, "TrainLabels", "*.nii.gz")))

train_files = [{"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)]
```
This part is to create dictionary that contains the paths of the training and validation data and labels.

```Python
original_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], a_min=-200, a_max=200,b_min=0.0, b_max=1.0, clip=True,), 
        ToTensord(keys=["image", "label"]),
    ]
)
```
Here we did only preprocessing just to show the difference between the original patient and the synthetic one.

------------------------------------------------------

If you want to read the whole article, you can visite [this link](https://pycad.co/3d-volumes-augmentation-for-tumor-segmentation/)

## An example of a synthetic patient

![Output image](https://github.com/amine0110/data-augmentation-for-3D-volumes/blob/main/example_generated_patient.PNG)


## 🆕 NEW

Learn how to effectively manage and process DICOM files in Python with our comprehensive course, designed to equip you with the skills and knowledge you need to succeed.

https://www.learn.pycad.co/course/dicom-simplified

