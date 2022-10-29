<!-- TITLE -->
## Glasses - A scalable CNN implementation for image classification (from scratch)
<div align="center">
<p align="justify">
Abstract—Machine learning model’s effectiveness is strictly
related to the data they are trained with, and the quality and
quantity of data directly impact the results of the produced
model. Dealing with the large amounts of data required for
some models is not trivial and requires specific programming
techniques and algorithms. In this paper, we will discuss a way
of training a feedforward neural network across a distributed
system to be able to deal with massive datasets. The proposed
solution utilizes the Mapreduce programming techniques; it
requires that small partitions of the dataset are processed on
copies of the neural networks in different machines of a cluster.
The outputs of the backpropagation phase of the training are
then used to update the original neural network. The process is
repeated till the network is trained.
</div>

<!-- ABOUT THE PROJECT -->
## About The Project
This project was developed for the university exams in Statistical Methods for Machine Learning and Algorithms for Massive Datasets.

Read the Paper: [Glasses](https://github.com/Ale-Ba2lero/Glasses-or-no-Glasses/blob/main/Glasses___A_scalable_CNN_implementation_for_image_classification__from_scratch_.pdf)


<!-- CONTACT -->
## Contact

Alessandro Ballerini - alessandroballerini95@gmail.com


<!-- GETTING STARTED -->
## Getting Started

Create a GCloud cluster. eg:
gcloud dataproc clusters create amd-cluster --enable-component-gateway --region europe-west6 
--zone europe-west6-a --master-machine-type c2-standard-8 --master-boot-disk-type pd-ssd 
--master-boot-disk-size 500 --num-workers 5 --worker-machine-type n1-standard-4 
--worker-boot-disk-size 500 --image-version 2.0-debian10 
--optional-components JUPYTER --project amd-cloud-324608


SSH forwarding (local):
gcloud beta compute ssh --zone "europe-west6-a" "amd-cluster-m"  --project "amd-cloud-324608" -- -L 8080:localhost:8080


Run Jupyter Notebook (remote):
jupyter notebook \
  --NotebookApp.allow_origin='https://colab.research.google.com' \
  --port=8080 \
  --NotebookApp.port_retries=0
