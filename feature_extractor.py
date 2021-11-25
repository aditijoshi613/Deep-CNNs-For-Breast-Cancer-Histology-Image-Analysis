{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "feature_extractor.py",
      "provenance": [],
      "authorship_tag": "ABX9TyOHjPlzed6PzGl1pqB6tR7l",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/aditijoshi613/Deep-CNNs-For-Breast-Cancer-Histology-Image-Analysis/blob/main/feature_extractor.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Y8cVH2YXka_C"
      },
      "source": [
        "#!/usr/bin/env python3\n",
        "\"\"\"Extract deep CNN features from a set of images and dump them as Numpy arrays image_file_name.npy\"\"\"\n",
        "\n",
        "import argparse\n",
        "import numpy as np\n",
        "import cv2\n",
        "from scipy import ndimage\n",
        "from os.path import basename, join, exists\n",
        "from os import makedirs\n",
        "from threaded_generator import threaded_generator\n",
        "from time import time\n",
        "import sys\n",
        "np.random.seed(13)\n",
        "\n",
        "PATCH_SIZES = [400, 650]\n",
        "SCALES = [0.5]\n",
        "\n",
        "DEFAULT_INPUT_DIR = \"data/train\"\n",
        "DEFAULT_PREPROCESSED_ROOT = \"data/preprocessed/train\"\n",
        "\n",
        "PATCHES_PER_IMAGE = 20\n",
        "AUGMENTATIONS_PER_IMAGE = 50\n",
        "COLOR_LO = 0.7\n",
        "COLOR_HI = 1.3\n",
        "BATCH_SIZE = 16     # decrease if necessary\n",
        "\n",
        "NUM_CACHED = 160\n",
        "\n",
        "\n",
        "def recursive_glob(root_dir, file_template=\"*.jpeg\"):\n",
        "    \"\"\"Traverse directory recursively. Starting with Python version 3.5, the glob module supports the \"**\" directive\"\"\"\n",
        "\n",
        "    if sys.version_info[0] * 10 + sys.version_info[1] < 35:\n",
        "        import fnmatch\n",
        "        import os\n",
        "        matches = []\n",
        "        for root, dirnames, filenames in os.walk(root_dir):\n",
        "            for filename in fnmatch.filter(filenames, file_template):\n",
        "                matches.append(os.path.join(root, filename))\n",
        "            print(fnmatch.filter(filenames, file_template))\n",
        "        return matches\n",
        "    else:\n",
        "        import glob\n",
        "        temp =  glob.glob(root_dir + \"/**/\" + file_template, recursive=True)\n",
        "        print(type(temp))\n",
        "        return temp\n",
        "\n",
        "\n",
        "def normalize_staining(img):\n",
        "    \"\"\"\n",
        "    Adopted from \"Classification of breast cancer histology images using Convolutional Neural Networks\",\n",
        "    Teresa Araújo , Guilherme Aresta, Eduardo Castro, José Rouco, Paulo Aguiar, Catarina Eloy, António Polónia,\n",
        "    Aurélio Campilho. https://doi.org/10.1371/journal.pone.0177544\n",
        "\n",
        "    Performs staining normalization.\n",
        "\n",
        "    # Arguments\n",
        "        img: Numpy image array.\n",
        "    # Returns\n",
        "        Normalized Numpy image array.\n",
        "    \"\"\"\n",
        "    Io = 240\n",
        "    beta = 0.15\n",
        "    alpha = 1\n",
        "    HERef = np.array([[0.5626, 0.2159],\n",
        "                      [0.7201, 0.8012],\n",
        "                      [0.4062, 0.5581]])\n",
        "    maxCRef = np.array([1.9705, 1.0308])\n",
        "\n",
        "    h, w, c = img.shape\n",
        "    img = img.reshape(h * w, c)\n",
        "    OD = -np.log((img.astype(\"uint16\") + 1) / Io)\n",
        "    ODhat = OD[(OD >= beta).all(axis=1)]\n",
        "    W, V = np.linalg.eig(np.cov(ODhat, rowvar=False))\n",
        "\n",
        "    Vec = -V.T[:2][::-1].T  # desnecessario o sinal negativo\n",
        "    That = np.dot(ODhat, Vec)\n",
        "    phi = np.arctan2(That[:, 1], That[:, 0])\n",
        "    minPhi = np.percentile(phi, alpha)\n",
        "    maxPhi = np.percentile(phi, 100 - alpha)\n",
        "    vMin = np.dot(Vec, np.array([np.cos(minPhi), np.sin(minPhi)]))\n",
        "    vMax = np.dot(Vec, np.array([np.cos(maxPhi), np.sin(maxPhi)]))\n",
        "    if vMin[0] > vMax[0]:\n",
        "        HE = np.array([vMin, vMax])\n",
        "    else:\n",
        "        HE = np.array([vMax, vMin])\n",
        "\n",
        "    HE = HE.T\n",
        "    Y = OD.reshape(h * w, c).T\n",
        "\n",
        "    C = np.linalg.lstsq(HE, Y)\n",
        "    maxC = np.percentile(C[0], 99, axis=1)\n",
        "\n",
        "    C = C[0] / maxC[:, None]\n",
        "    C = C * maxCRef[:, None]\n",
        "    Inorm = Io * np.exp(-np.dot(HERef, C))\n",
        "    Inorm = Inorm.T.reshape(h, w, c).clip(0, 255).astype(\"uint8\")\n",
        "\n",
        "    return Inorm\n",
        "\n",
        "\n",
        "def hematoxylin_eosin_aug(img, low=0.7, high=1.3, seed=None):\n",
        "    \"\"\"\n",
        "    \"Quantification of histochemical staining by color deconvolution\"\n",
        "    Arnout C. Ruifrok, Ph.D. and Dennis A. Johnston, Ph.D.\n",
        "    http://www.math-info.univ-paris5.fr/~lomn/Data/2017/Color/Quantification_of_histochemical_staining.pdf\n",
        "\n",
        "    Performs random hematoxylin-eosin augmentation\n",
        "\n",
        "    # Arguments\n",
        "        img: Numpy image array.\n",
        "        low: Low boundary for augmentation multiplier\n",
        "        high: High boundary for augmentation multiplier\n",
        "    # Returns\n",
        "        Augmented Numpy image array.\n",
        "    \"\"\"\n",
        "    D = np.array([[1.88, -0.07, -0.60],\n",
        "                  [-1.02, 1.13, -0.48],\n",
        "                  [-0.55, -0.13, 1.57]])\n",
        "    M = np.array([[0.65, 0.70, 0.29],\n",
        "                  [0.07, 0.99, 0.11],\n",
        "                  [0.27, 0.57, 0.78]])\n",
        "    Io = 240\n",
        "\n",
        "    h, w, c = img.shape\n",
        "    OD = -np.log10((img.astype(\"uint16\") + 1) / Io)\n",
        "    C = np.dot(D, OD.reshape(h * w, c).T).T\n",
        "    r = np.ones(3)\n",
        "    r[:2] = np.random.RandomState(seed).uniform(low=low, high=high, size=2)\n",
        "    img_aug = np.dot(C, M) * r\n",
        "\n",
        "    img_aug = Io * np.exp(-img_aug * np.log(10)) - 1\n",
        "    img_aug = img_aug.reshape(h, w, c).clip(0, 255).astype(\"uint8\")\n",
        "    return img_aug\n",
        "\n",
        "\n",
        "def zoom_aug(img, zoom_var, seed=None):\n",
        "    \"\"\"Performs a random spatial zoom of a Numpy image array.\n",
        "\n",
        "    # Arguments\n",
        "        img: Numpy image array.\n",
        "        zoom_var: zoom range multiplier for width and height.\n",
        "        seed: Random seed.\n",
        "    # Returns\n",
        "        Zoomed Numpy image array.\n",
        "    \"\"\"\n",
        "    scale = np.random.RandomState(seed).uniform(low=1 / zoom_var, high=zoom_var)\n",
        "    resized_img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)\n",
        "    return resized_img\n",
        "\n",
        "\n",
        "def get_crops(img, size, n, seed=None):\n",
        "    \"\"\"Creates random square crops of given size from a Numpy image array. No rotation added\n",
        "\n",
        "    # Arguments\n",
        "        img: Numpy image array.\n",
        "        size: size of crops.\n",
        "        n: number of crops\n",
        "        seed: Random seed.\n",
        "    # Returns\n",
        "        Numpy array of crops, shape (n, size, size, c).\n",
        "    \"\"\"\n",
        "    h, w, c = img.shape\n",
        "    assert all([size < h, size < w])\n",
        "    crops = []\n",
        "    for _ in range(n):\n",
        "        top = np.random.randint(low=0, high=h - size + 1)\n",
        "        left = np.random.randint(low=0, high=w - size + 1)\n",
        "        crop = img[top: top + size, left: left + size].copy()\n",
        "        crop = np.rot90(crop, np.random.randint(low=0, high=4))\n",
        "        if np.random.random() > 0.5:\n",
        "            crop = np.flipud(crop)\n",
        "        if np.random.random() > 0.5:\n",
        "            crop = np.fliplr(crop)\n",
        "        crops.append(crop)\n",
        "\n",
        "    crops = np.stack(crops)\n",
        "    assert crops.shape == (n, size, size, c)\n",
        "    return crops\n",
        "\n",
        "\n",
        "def get_crops_free(img, size, n, seed=None):\n",
        "    \"\"\"Creates random square crops of given size from a Numpy image array. With rotation\n",
        "\n",
        "    # Arguments\n",
        "        img: Numpy image array.\n",
        "        size: size of crops.\n",
        "        n: number of crops\n",
        "        seed: Random seed.\n",
        "    # Returns\n",
        "        Numpy array of crops, shape (n, size, size, c).\n",
        "    \"\"\"\n",
        "    h, w, c = img.shape\n",
        "    assert all([size < h, size < w])\n",
        "    d = int(np.ceil(size / np.sqrt(2)))\n",
        "    crops = []\n",
        "    for _ in range(n):\n",
        "        center_y = np.random.randint(low=0, high=h - size + 1) + size // 2\n",
        "        center_x = np.random.randint(low=0, high=w - size + 1) + size // 2\n",
        "        m = min(center_y, center_x, h - center_y, w - center_x)\n",
        "        if m < d:\n",
        "            max_angle = np.pi / 4 - np.arccos(m / d)\n",
        "            top = center_y - m\n",
        "            left = center_x - m\n",
        "            precrop = img[top: top + 2 * m, left: left + 2 * m]\n",
        "        else:\n",
        "            max_angle = np.pi / 4\n",
        "            top = center_y - d\n",
        "            left = center_x - d\n",
        "            precrop = img[top: top + 2 * d, left: left + 2 * d]\n",
        "\n",
        "        precrop = np.rot90(precrop, np.random.randint(low=0, high=4))\n",
        "        angle = np.random.uniform(low=-max_angle, high=max_angle)\n",
        "        precrop = ndimage.rotate(precrop, angle * 180 / np.pi, reshape=False)\n",
        "\n",
        "        precrop_h, precrop_w, _ = precrop.shape\n",
        "        top = (precrop_h - size) // 2\n",
        "        left = (precrop_w - size) // 2\n",
        "        crop = precrop[top: top + size, left: left + size]\n",
        "\n",
        "        if np.random.random() > 0.5:\n",
        "            crop = np.flipud(crop)\n",
        "        if np.random.random() > 0.5:\n",
        "            crop = np.fliplr(crop)\n",
        "        crops.append(crop)\n",
        "\n",
        "    crops = np.stack(crops)\n",
        "    assert crops.shape == (n, size, size, c)\n",
        "    return crops\n",
        "\n",
        "\n",
        "def norm_pool(features, p=3):\n",
        "    \"\"\"Performs descriptor pooling\n",
        "\n",
        "    # Arguments\n",
        "        features: Numpy array of descriptors.\n",
        "        p: degree of pooling.\n",
        "    # Returns\n",
        "        Numpy array of pooled descriptor.\n",
        "    \"\"\"\n",
        "    return np.power(np.power(features, p).mean(axis=0), 1/p)\n",
        "\n",
        "\n",
        "def encode(crops, model):\n",
        "    \"\"\"Encodes crops\n",
        "\n",
        "    # Arguments\n",
        "        crops: Numpy array of crops.\n",
        "        model: Keras encoder.\n",
        "    # Returns\n",
        "        Numpy array of pooled descriptor.\n",
        "    \"\"\"\n",
        "    features = model.predict(crops)\n",
        "    pooled_features = norm_pool(features)\n",
        "    return pooled_features\n",
        "\n",
        "\n",
        "def process_image(image_file):  #imp\n",
        "    \"\"\"Extract multiple crops from a single image\n",
        "\n",
        "    # Arguments\n",
        "        image_file: Path to image.\n",
        "    # Yields\n",
        "        Numpy array of image crops.\n",
        "    \"\"\"\n",
        "    img = cv2.imread(image_file)\n",
        "    if SCALE != 1:\n",
        "        img = cv2.resize(img, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)\n",
        "    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\n",
        "    #print('meee',img.shape)\n",
        "    #img_norm = normalize_staining(img)\n",
        "    img_norm = img\n",
        "  #  print('youu', img_norm.shape)\n",
        "    for _ in range(AUGMENTATIONS_PER_IMAGE):\n",
        "        #img_aug = hematoxylin_eosin_aug(img_norm, low=COLOR_LO, high=COLOR_HI)\n",
        "        # img_aug = zoom_aug(img_aug, ZOOM_VAR)\n",
        "        img_aug = img_norm\n",
        "        #print('mee', img_aug.shape)\n",
        "\n",
        "        # single_image_crops = get_crops_free(img_aug, PATCH_SZ, PATCHES_PER_IMAGE)\n",
        "        single_image_crops = get_crops(img_aug, PATCH_SZ, PATCHES_PER_IMAGE)\n",
        "        #print('them', type(single_image_crops), single_image_crops.shape)\n",
        "        yield single_image_crops\n",
        "\n",
        "\n",
        "def crops_gen(file_list):   #imp\n",
        "    \"\"\"Generates batches of crops from image list, one augmentation a time\n",
        "\n",
        "    # Arguments\n",
        "        file_list: List of image files.\n",
        "    # Yields\n",
        "        Tuple of Numpy array of image crops and name of the file.\n",
        "    \"\"\"\n",
        "    for i, (image_file, output_file) in enumerate(file_list):\n",
        "        print(\"Crops generator:\", i + 1)\n",
        "        for crops in process_image(image_file):\n",
        "            yield crops, output_file\n",
        "\n",
        "\n",
        "def features_gen(crops_and_output_file, model):\n",
        "    \"\"\"Processes crop generator, encodes them and dumps pooled descriptors\n",
        "\n",
        "    # Arguments\n",
        "        crops_and_output_file: generator of crops and file names.\n",
        "        model: Keras encoder.\n",
        "    # Returns: None\n",
        "    \"\"\"\n",
        "    ts = time()\n",
        "    current_file = None\n",
        "    pooled_features = []\n",
        "    i = 0\n",
        "    for j, (crops, output_file) in enumerate(crops_and_output_file):\n",
        "        if current_file is None:\n",
        "            current_file = output_file\n",
        "        features = encode(crops, model)\n",
        "        if output_file == current_file:\n",
        "            pooled_features.append(features)\n",
        "        else:\n",
        "            np.save(current_file, np.stack(pooled_features))\n",
        "            pooled_features = [features]\n",
        "            current_file = output_file\n",
        "            average_time = int((time() - ts) / (i + 1))\n",
        "            print(\"Feature generator: {}, {} sec/image.\".format(i + 1, average_time))\n",
        "            i += 1\n",
        "    if len(pooled_features) > 0:\n",
        "        np.save(current_file, np.stack(pooled_features))\n",
        "\n",
        "\n",
        "if __name__ == \"__main__\":\n",
        "    parser = argparse.ArgumentParser()\n",
        "    arg = parser.add_argument\n",
        "    arg(\"--images\",\n",
        "        required=False,\n",
        "        default=DEFAULT_INPUT_DIR,\n",
        "        metavar=\"img_dir\",\n",
        "        help=\"Input image directory. Default: data/train\")\n",
        "    arg(\"--features\",\n",
        "        required=False,\n",
        "        default=DEFAULT_PREPROCESSED_ROOT,\n",
        "        metavar=\"feat_dir\",\n",
        "        help=\"Feature root dir. Default: data/preprocessed/train\")\n",
        "    args = parser.parse_args()\n",
        "    INPUT_DIR = args.images\n",
        "    PREPROCESSED_ROOT = args.features\n",
        "\n",
        "    from models import ResNet, Inception, VGG\n",
        "    NN_MODELS = [ResNet, Inception, VGG]\n",
        "\n",
        "    input_files = recursive_glob(INPUT_DIR)\n",
        "    #print(INPUT_DIR)\n",
        "    #print(input_files)\n",
        "\n",
        "    for SCALE in SCALES:\n",
        "        print(\"SCALE:\", SCALE)\n",
        "        for NN_MODEL in NN_MODELS:\n",
        "            print(\"NN_MODEL:\", NN_MODEL.__name__)\n",
        "            for PATCH_SZ in PATCH_SIZES:\n",
        "                print(\"PATCH_SZ:\", PATCH_SZ)\n",
        "                PREPROCESSED_PATH = join(PREPROCESSED_ROOT, \"{}-{}-{}\".format(NN_MODEL.__name__, SCALE, PATCH_SZ))\n",
        "                print(PREPROCESSED_PATH)\n",
        "                if not exists(PREPROCESSED_PATH):\n",
        "                    makedirs(PREPROCESSED_PATH)\n",
        "\n",
        "                model = NN_MODEL(batch_size=BATCH_SIZE)\n",
        "\n",
        "                output_files = [(join(PREPROCESSED_PATH, basename(f).replace(\"jpeg\", \"npy\"))) if 'jpeg' in f else (join(PREPROCESSED_PATH, basename(f).replace(\"jpg\", \"npy\"))) for f in input_files]\n",
        "                file_list = zip(input_files, output_files)\n",
        "                # print(*[(i,j)  for i,j in file_list], sep = \"\\n\")\n",
        "                crops_and_output_file = crops_gen(file_list)\n",
        "                crops_and_output_file_ = threaded_generator(crops_and_output_file, num_cached=NUM_CACHED)\n",
        "                features_gen(crops_and_output_file_, model)\n",
        "                print('done ' + \"{}-{}-{}\".format(NN_MODEL.__name__, SCALE, PATCH_SZ))"
      ],
      "execution_count": null,
      "outputs": []
    }
  ]
}