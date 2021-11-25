{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "run_svm.py",
      "provenance": [],
      "authorship_tag": "ABX9TyMtXUy1zJEEA3/AqSenqHF8",
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
        "<a href=\"https://colab.research.google.com/github/aditijoshi613/Deep-CNNs-For-Breast-Cancer-Histology-Image-Analysis/blob/main/run_svm.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "vVWclK6gkvPU"
      },
      "source": [
        "#!/usr/bin/env python3\n",
        "\"\"\"Trains SVM models on various features, data splits. Dumps models and predictions\"\"\"\n",
        "\n",
        "import pickle\n",
        "import numpy as np\n",
        "import lightgbm as lgb\n",
        "from sklearn.metrics import accuracy_score\n",
        "from utils import load_data\n",
        "from os.path import join, exists\n",
        "from os import makedirs\n",
        "import argparse\n",
        "\n",
        "\n",
        "CROP_SIZES = [400, 650]\n",
        "SCALES = [0.5]\n",
        "NN_MODELS = [\"ResNet\", \"Inception\", \"VGG\"]\n",
        "\n",
        "AUGMENTATIONS_PER_IMAGE = 50\n",
        "NUM_CLASSES = 4\n",
        "RANDOM_STATE = 1\n",
        "N_SEEDS = 5                                       #is it really requires in svm??\n",
        "VERBOSE_EVAL = -1 #True #False                    #MY CHANGE\n",
        "with open(\"data/folds-10.pkl\", \"rb\") as f:\n",
        "    FOLDS = pickle.load(f)\n",
        "\n",
        "LGBM_MODELS_ROOT = \"models/svms\"\n",
        "CROSSVAL_PREDICTIONS_ROOT = \"predictions\"\n",
        "DEFAULT_PREPROCESSED_ROOT = \"data/preprocessed/train\"\n",
        "\n",
        "\n",
        "def _mean(x, mode=\"arithmetic\"):\n",
        "    \"\"\"\n",
        "    Calculates mean probabilities across augmented data\n",
        "\n",
        "    # Arguments\n",
        "        x: Numpy 3D array of probability scores, (N, AUGMENTATIONS_PER_IMAGE, NUM_CLASSES)\n",
        "        mode: type of averaging, can be \"arithmetic\" or \"geometric\"\n",
        "    # Returns\n",
        "        Mean probabilities 2D array (N, NUM_CLASSES)\n",
        "    \"\"\"\n",
        "    assert mode in [\"arithmetic\", \"geometric\"]\n",
        "    if mode == \"arithmetic\":\n",
        "        x_mean = x.mean(axis=1)\n",
        "    else:\n",
        "        x_mean = np.exp(np.log(x + 1e-7).mean(axis=1))\n",
        "        x_mean = x_mean / x_mean.sum(axis=1, keepdims=True)\n",
        "    return x_mean\n",
        "\n",
        "\n",
        "if __name__ == \"__main__\":\n",
        "    parser = argparse.ArgumentParser()\n",
        "    arg = parser.add_argument\n",
        "    arg(\"--features\",\n",
        "        required=False,\n",
        "        default=DEFAULT_PREPROCESSED_ROOT,\n",
        "        metavar=\"feat_dir\",\n",
        "        help=\"Feature root dir. Default: data/preprocessed/train\")    #input\n",
        "    args = parser.parse_args()\n",
        "    PREPROCESSED_ROOT = args.features\n",
        "\n",
        "    print('meee' , PREPROCESSED_ROOT)\n",
        "\n",
        "    learning_rate = 0.1\n",
        "    num_round = 70\n",
        "    # param = {\n",
        "    #     \"objective\": \"multiclass\",\n",
        "    #     \"num_class\": NUM_CLASSES,\n",
        "    #     \"metric\": [\"multi_logloss\", \"multi_error\"],\n",
        "    #     \"verbose\": 0,\n",
        "    #     \"learning_rate\": learning_rate,\n",
        "    #     \"num_leaves\": 191,\n",
        "    #     \"feature_fraction\": 0.46,\n",
        "    #     \"bagging_fraction\": 0.69,\n",
        "    #     \"bagging_freq\": 0,\n",
        "    #     \"max_depth\": 7,\n",
        "    # }\n",
        "\n",
        "    params_grid = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],\n",
        "                    'C': [1, 10, 100, 1000]},\n",
        "                   {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]\n",
        "\n",
        "\n",
        "    for SCALE in SCALES:\n",
        "        print(\"SCALE:\", SCALE)\n",
        "        for NN_MODEL in NN_MODELS:\n",
        "            print(\"NN_MODEL:\", NN_MODEL)\n",
        "            for CROP_SZ in CROP_SIZES:\n",
        "                #print(\"PATCH_SZ:\", CROP_SZ)\n",
        "                INPUT_DIR = join(PREPROCESSED_ROOT, \"{}-{}-{}\".format(NN_MODEL, SCALE, CROP_SZ))\n",
        "                acc_all_seeds = []\n",
        "                for seed in range(N_SEEDS):\n",
        "                    accuracies = []\n",
        "\n",
        "                    # print(len(FOLDS))          #flod has data of what models to train and test with at\n",
        "                                                 #what fold, also what thier classes are.\n",
        "                    # print('fold_p', FOLDS)\n",
        "                    # print('p')\n",
        "\n",
        "                    for fold in range(len(FOLDS)):\n",
        "                        feature_fraction_seed = RANDOM_STATE + seed * 10 + fold\n",
        "                        bagging_seed = feature_fraction_seed + 1\n",
        "                        param.update({\"feature_fraction_seed\": feature_fraction_seed, \"bagging_seed\": bagging_seed})\n",
        "                        print(\"Fold {}/{}, seed {}\".format(fold + 1, len(FOLDS), seed))\n",
        "                        x_train, y_train, x_test, y_test = load_data(INPUT_DIR, FOLDS, fold)\n",
        "                        # print(x_train[1])\n",
        "                        # print(x_train[1].shape)\n",
        "                        # print(y_train)\n",
        "                        #print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)\n",
        "                        # print(x_test)\n",
        "                        # print(y_test)\n",
        "                #\n",
        "                #\n",
        "                        train_data = lgb.Dataset(x_train, label=y_train)\n",
        "                        test_data = lgb.Dataset(x_test, label=y_test)\n",
        "                        gbm = lgb.train(param, train_data, num_round, valid_sets=[test_data], verbose_eval=VERBOSE_EVAL)\n",
        "                #\n",
        "                #\n",
        "                #\n",
        "                #         # pickle model\n",
        "                #         model_file = \"lgbm-{}-{}-{}-f{}-s{}.pkl\".format(NN_MODEL, SCALE, CROP_SZ, fold, seed)\n",
        "                #         model_root = join(LGBM_MODELS_ROOT, NN_MODEL)\n",
        "                #         if not exists(model_root):\n",
        "                #             makedirs(model_root)\n",
        "                #         with open(join(model_root, model_file), \"wb\") as f:\n",
        "                #             pickle.dump(gbm, f)\n",
        "                #\n",
        "                #         scores = gbm.predict(x_test)\n",
        "                #         scores = scores.reshape(-1, AUGMENTATIONS_PER_IMAGE, NUM_CLASSES)\n",
        "                #         preds = {\n",
        "                #             \"files\": FOLDS[fold][\"test\"][\"x\"],\n",
        "                #             \"y_true\": y_test,\n",
        "                #             \"scores\": scores,\n",
        "                #         }\n",
        "                #         preds_file = \"lgbm_preds-{}-{}-{}-f{}-s{}.pkl\".format(NN_MODEL, SCALE, CROP_SZ,\n",
        "                #                                                               fold, seed)\n",
        "                #         preds_root = join(CROSSVAL_PREDICTIONS_ROOT, NN_MODEL)\n",
        "                #         if not exists(preds_root):\n",
        "                #             makedirs(preds_root)\n",
        "                #         with open(join(preds_root, preds_file), \"wb\") as f:\n",
        "                #             pickle.dump(preds, f)                    #makes files if not present\n",
        "                #\n",
        "                #         mean_scores = _mean(scores, mode=\"arithmetic\")\n",
        "                #         y_pred = np.argmax(mean_scores, axis=1)\n",
        "                #         y_true = y_test[::AUGMENTATIONS_PER_IMAGE]\n",
        "                #         acc = accuracy_score(y_true, y_pred)\n",
        "                #         print(\"Accuracy:\", acc)\n",
        "                #         accuracies.append(acc)\n",
        "                #\n",
        "                #     acc_seed = np.array(accuracies).mean()  # acc of a seed\n",
        "                #     acc_all_seeds.append(acc_seed)\n",
        "                #     print(\"{}-{}-{} Accuracies: [{}], mean {:5.3}\".format(NN_MODEL, SCALE, CROP_SZ,\n",
        "                #                                                           \", \".join(map(lambda s: \"{:5.3}\".format(s), accuracies)),\n",
        "                #                                                           acc_seed))\n",
        "                # print(\"Accuracy of all seeds {:5.3}\".format(np.array(acc_all_seeds).mean()))\n",
        "\n",
        "\"\"\"\n",
        "ResNet-1.0-800\n",
        "learning_rate = 0.1, 70 steps\n",
        "* new acc reached, loss 0.63, acc 0.825, knobs num_leaves 191, feature_fraction 0.46, bagging_fraction 0.66, max_depth 7\n",
        "[0.875, 0.725, 0.875, 0.875,   0.8,  0.75,   0.8,  0.85,  0.85,  0.85], mean 0.825\n",
        "[0.875, 0.775, 0.825, 0.875, 0.775,  0.75,   0.8,  0.85,  0.85, 0.825], mean  0.82\n",
        "[0.875,  0.75,  0.85,  0.85,   0.8,  0.75, 0.725, 0.875, 0.875, 0.875], mean 0.823\n",
        "[0.875,  0.75,  0.85,  0.85,   0.8, 0.775, 0.775, 0.825,  0.85, 0.825], mean 0.817\n",
        "[0.875,  0.75, 0.875, 0.875, 0.775, 0.775, 0.825,  0.85,   0.9,  0.85], mean 0.835\n",
        "Accuracy of all seeds 0.824\n",
        "\n",
        "ResNet-1.0-1300\n",
        "learning_rate = 0.1, 60 steps [0.9, 0.775, 0.825, 0.875, 0.8, 0.775, 0.8, 0.875, 0.85, 0.85]\n",
        "* new acc reached, loss 0.63, acc 0.825, knobs num_leaves 191, feature_fraction 0.46, bagging_fraction 0.66, max_depth 7\n",
        "[0.925,  0.75,  0.85,  0.85,   0.8, 0.725,   0.8, 0.825,   0.9,  0.85], mean 0.828\n",
        "[0.875,  0.75,   0.8, 0.875,  0.75, 0.725,   0.8, 0.825, 0.875,  0.85], mean 0.812\n",
        "[0.875,  0.75,   0.8, 0.825, 0.775, 0.725, 0.825,  0.85,  0.95, 0.875], mean 0.825\n",
        "[  0.9, 0.775,   0.8,  0.85, 0.725, 0.725,   0.8,   0.8, 0.875, 0.825], mean 0.807\n",
        "[ 0.85, 0.725,   0.8, 0.825, 0.725,  0.75,   0.8,   0.8, 0.875,  0.85], mean   0.8\n",
        "Accuracy of all seeds 0.815\n",
        "\n",
        "VGG-1.0-800\n",
        "learning_rate = 0.1, 70 steps\n",
        "* new acc reached, loss 0.63, acc 0.825, knobs num_leaves 191, feature_fraction 0.46, bagging_fraction 0.66, max_depth 7\n",
        "[ 0.85, 0.775, 0.825, 0.875,  0.85, 0.775,   0.8,   0.8,  0.85,   0.8], mean  0.82\n",
        "[ 0.85,  0.75,  0.85, 0.875, 0.825, 0.775, 0.775, 0.825, 0.875, 0.775], mean 0.818\n",
        "[ 0.85,  0.75, 0.825, 0.875,  0.85,  0.75, 0.825, 0.825, 0.875,  0.75], mean 0.818\n",
        "[ 0.85,   0.8, 0.825, 0.875, 0.825,   0.8,   0.8,   0.8, 0.875, 0.775], mean 0.823\n",
        "[0.825, 0.775, 0.775, 0.875,  0.85, 0.775,   0.8,   0.8,  0.85, 0.725], mean 0.805\n",
        "Accuracy of all seeds 0.816\n",
        "\n",
        "VGG-1.0-1300\n",
        "learning_rate = 0.1, 70 steps\n",
        "* new acc reached, loss 0.63, acc 0.825, knobs num_leaves 191, feature_fraction 0.46, bagging_fraction 0.66, max_depth 7\n",
        "[0.825,  0.75,   0.8, 0.875, 0.825,   0.8,   0.8,  0.75,  0.85,   0.8], mean 0.807\n",
        "[0.825,  0.75, 0.825,   0.9,   0.8, 0.825, 0.725, 0.825, 0.875, 0.775], mean 0.812\n",
        "[ 0.85,   0.8,   0.8, 0.875, 0.775, 0.775, 0.775, 0.775, 0.875, 0.775], mean 0.808\n",
        "[  0.8, 0.775, 0.775,   0.9,   0.8,   0.8,   0.8, 0.825, 0.875, 0.775], mean 0.812\n",
        "[0.825, 0.825,   0.8,   0.9,   0.8,   0.8, 0.725,   0.8,  0.85,   0.8], mean 0.812\n",
        "Accuracy of all seeds  0.81\n",
        "\n",
        "\n",
        "ResNet-0.5-650\n",
        "learning_rate = 0.1, 70 steps\n",
        "* new acc reached, loss 0.63, acc 0.825, knobs num_leaves 191, feature_fraction 0.46, bagging_fraction 0.66, max_depth 7\n",
        "[ 0.95, 0.775, 0.875,   0.9, 0.825, 0.775,  0.85,  0.85,  0.85, 0.825], mean 0.847\n",
        "[  0.9, 0.775, 0.875,   0.9, 0.775,   0.7, 0.875,   0.8,  0.85, 0.825], mean 0.828\n",
        "[  0.9, 0.775,  0.85,   0.9, 0.825,  0.75, 0.875, 0.825, 0.825, 0.825], mean 0.835\n",
        "[0.875, 0.775,  0.85, 0.875,   0.8, 0.725,  0.85, 0.825,  0.85, 0.825], mean 0.825\n",
        "[0.925, 0.775,  0.85,   0.9, 0.825,  0.75, 0.825,  0.85,  0.85, 0.825], mean 0.838\n",
        "Accuracy of all seeds 0.834\n",
        "\n",
        "\n",
        "ResNet-0.5-400\n",
        "learning_rate = 0.1, 70 steps\n",
        "* new acc reached, loss 0.63, acc 0.825, knobs num_leaves 191, feature_fraction 0.46, bagging_fraction 0.66, max_depth 7\n",
        "[0.925, 0.825, 0.875, 0.875, 0.775, 0.825, 0.825, 0.825, 0.825, 0.825], mean 0.84\n",
        "[0.925, 0.775, 0.875, 0.875,   0.8,  0.85,  0.85,   0.8,  0.85, 0.825], mean 0.842\n",
        "[  0.9, 0.725, 0.875, 0.875, 0.825,  0.85, 0.875,  0.85,  0.85, 0.825], mean 0.845\n",
        "[0.925, 0.775, 0.875, 0.875,   0.8,  0.85,  0.85,  0.85,  0.85, 0.825], mean 0.847\n",
        "[0.925, 0.775, 0.825, 0.875, 0.775, 0.825,  0.85, 0.825, 0.825, 0.825], mean 0.833\n",
        "acc_all_seeds 0.841\n",
        "\n",
        "\n",
        "VGG-0.5-400\n",
        "learning_rate = 0.1, 70 steps\n",
        "* new acc reached, loss 0.63, acc 0.825, knobs num_leaves 191, feature_fraction 0.46, bagging_fraction 0.66, max_depth 7  \n",
        "[ 0.85,  0.85,   0.8, 0.825, 0.825,   0.8, 0.825,  0.85, 0.875, 0.825], mean 0.832\n",
        "[0.875, 0.825,  0.85,  0.85, 0.875,  0.85,   0.8, 0.825, 0.875, 0.825], mean 0.845\n",
        "[  0.9, 0.825, 0.825,  0.85,  0.85,   0.8,   0.8,  0.85, 0.875,  0.85], mean 0.842\n",
        "[0.875, 0.825,   0.8,  0.85, 0.825, 0.875,   0.8, 0.775, 0.875, 0.825], mean 0.832\n",
        "[0.875, 0.825,   0.8, 0.825, 0.825,   0.8,   0.8,   0.8, 0.875, 0.825], mean 0.825\n",
        "acc_all_seeds 0.835\n",
        "\n",
        "VGG-0.5-650\n",
        "learning_rate = 0.1, 70 steps\n",
        "* new acc reached, loss 0.63, acc 0.825, knobs num_leaves 191, feature_fraction 0.46, bagging_fraction 0.66, max_depth 7\n",
        "[  0.9, 0.825, 0.775,  0.85, 0.825,   0.8, 0.825, 0.875,   0.9,   0.8], mean 0.838\n",
        "[  0.9, 0.825, 0.775,  0.85,   0.8,   0.8,   0.8, 0.875,   0.9, 0.775], mean  0.83\n",
        "[  0.9, 0.875, 0.825,  0.85,   0.8,  0.75,   0.8, 0.875,   0.9, 0.825], mean  0.84\n",
        "[0.875,  0.85,  0.75,  0.85,   0.8,  0.75, 0.825,   0.8, 0.875,   0.8], mean 0.818\n",
        "[  0.9,   0.9,   0.8,  0.85, 0.825,   0.8, 0.825,  0.85, 0.875, 0.825], mean 0.845\n",
        "acc_all_seeds 0.8342\n",
        "\n",
        "\n",
        "Inception_adv-1.0-1300\n",
        "best acc reached so far 0.67, acc 0.7875, knobs num_leaves 121, feature_fraction 0.34, bagging_fraction 0.69, max_depth 38\n",
        "learning_rate = 0.1, 60 steps [0.8, 0.85, 0.7375, 0.7875, 0.7375]\n",
        "learning_rate = 0.1, 60 steps [0.8, 0.85, 0.8, 0.875, 0.8, 0.75, 0.75, 0.8, 0.8, 0.65]\n",
        "\n",
        "Inception-0.5-650\n",
        "* new acc reached, loss 0.63, acc 0.825, knobs num_leaves 191, feature_fraction 0.46, bagging_fraction 0.66, max_depth 7\n",
        "[  0.9, 0.825,  0.75,   0.9,  0.85,   0.8,   0.8,  0.85, 0.775, 0.775], mean 0.823\n",
        "[0.925,   0.8, 0.725,   0.9,  0.85,   0.8, 0.825, 0.825,   0.8,  0.75], mean  0.82\n",
        "[  0.9, 0.825,  0.75,   0.9, 0.825, 0.825,  0.85,  0.85, 0.775, 0.775], mean 0.828\n",
        "[0.925,   0.9, 0.725,   0.9, 0.825, 0.825, 0.825,  0.85,  0.75,  0.75], mean 0.828\n",
        "[  0.9, 0.875, 0.725,   0.9,  0.85,   0.8,   0.8,  0.85,   0.8,   0.8], mean  0.83\n",
        "Accuracy of all seeds 0.826\n",
        "\n",
        "Inception-0.5-400\n",
        "* new acc reached, loss 0.63, acc 0.825, knobs num_leaves 191, feature_fraction 0.46, bagging_fraction 0.66, max_depth 7\n",
        "(no data)\n",
        "\"\"\""
      ],
      "execution_count": null,
      "outputs": []
    }
  ]
}