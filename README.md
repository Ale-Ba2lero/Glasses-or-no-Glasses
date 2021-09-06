# CNN-FromScratch

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
