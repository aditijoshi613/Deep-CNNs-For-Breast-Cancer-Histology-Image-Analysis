Deep Convolutional Neural Networks for Breast Cancer Histology
Course Project | Guide: Prof. Biplab Banerjee | Machine Learning For Remote Sensing- II

Team Members
1.	Prakash Prasad
2.	Aditi Ganesh Joshi
3.	Neha Jahnavi
Original Paper: @article{rakhlin2018deep,title={Deep Convolutional Neural Networks for Breast Cancer Histology Image Analysis},author={Rakhlin, Alexander and Shvets, Alexey and Iglovikov, Vladimir and Kalinin, Alexandr A},journal={arXiv preprint arXiv:1802.00752},year={2018}}- https://github.com/alexander-rakhlin/ICIAR2018
Presentation https://docs.google.com/presentation/d/1gfamCcPKf0OAzCFmkyPYjfg0F0ntA4CWlExavzHrN4Y/edit#slide=id.p
Video
https://drive.google.com/file/d/1ScwTMMey0wxIfGeqSPDXh2NMun7tnw77/view


We first use preprocessed features and see accuracy, next the same model is applied on COVID-19 CT-Scan image dataset in order to check how well the transfer learning model works. Finally, we try to use multiclass SVM and replace Light GBM and verify accuracy.

Code execution for running with pre-trained features:
●	Add folder ICIAR-2018 in My Drive. 
●	Open “Deep Convolutional Neural Networks for Breast Cancer Histology: Main code file.ipynb” on Google Colab and Run. 
This code downloads preprocessed features of images, runs it on Light GBM classifier and gives 10-fold cross validation accuracy


Extracting features from COVID-19 images: 
●	Download “COVID2020-master” from “covid and svm” folder
●	Run Feature_extractor.py in ide with cpu or gpu 
●	The code preprocesses covid images from “\Data\train”  and returns npy files
●	.npy files for each cnn model for each crop size [400, 600]  are stored in “data\preprocessed\"{}-{}-{}".format(NN_MODEL.__name__, 0.5, PATCH_SZ)”
●	Note that argparse module doesn’t allow running on ipynb kernels
●	ETA on CPU : 7days 2 hours 

Running preprocessed features on multiclass SVM (*bugs present):
●	Download “ICAR2018-master” from “covid and svm” folder
●	Run run_svm.py on cpu. 
●	Bug present 

File description: 
Main code file.ipynb: runs preprocessed features on lightgbm 
ICIAR-2018 : 
●	Download models.py - downloads pretrained features and weights 
●	Crossvalidate_blending.py - finds 10-fold cross validation accuracy on presaved lgbm weights 
Covid: 
●	Feature extractor.py extracts features 
Svm - 
●	Run_svm - runs preprocessed features on svm





