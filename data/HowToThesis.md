# HowToThesis

## Google cloud
### Quick Overview

Command group	            Description
gcloud compute	            Commands related to Compute Engine in general availability
gcloud compute instances	Commands related to Compute Engine instances in general availability
gcloud beta compute	        Commands related to Compute Engine in Beta
gcloud alpha app	        Commands related to managing App Engine deployments in Alpha

- Connect to my instance
gcloud compute --project "deep-learning-media-filter" ssh --zone "us-central1-c" "instance-1"

## TensorFlow

- Arrancar virtualenv: source activate tensorflow
- Salir de una sesion de virtualenv: deactivate
- Arrancar tensorboard: tensorboard --logdir tf_logs/