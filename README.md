# Sentinel-Classification
Land Use and Land Cover Classification with Sentinel-2

![alt text](images/overview.jpg "overview image")

This project deals with the problem of classifying land use and land cover using Sentinel-2 satellite images.
The data used to train the models is the EuroSAT dataset which can be downloaded here : [EuroSAT Dataset](http://madm.dfki.de/files/sentinel/EuroSAT.zip)

The data was split into train and validation with the latter having a size of 20% out of the whole dataset.

A pretrained Resnet50 model was trained on 80% of the data for 20 epochs, and achieved an overall accuracy of 98.3% on the validation set.


## References : 

[1] Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification. Patrick Helber, Benjamin Bischke, Andreas Dengel, Damian Borth. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 2019.

[2] Introducing EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification. Patrick Helber, Benjamin Bischke, Andreas Dengel. 2018 IEEE International Geoscience and Remote Sensing Symposium, 2018.
