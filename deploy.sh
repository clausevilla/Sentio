# Author: Lian Shi
# File to deploy, it will run build image, push to Google cloud artifact registry, restart Kubernetes pods


#!/bin/bash
echo "(1/4)Building image..."
docker build --platform linux/amd64 -t europe-north2-docker.pkg.dev/sentio1/sentio/sentio-web:latest .

echo "(2/4)Pushing image..."
docker push europe-north2-docker.pkg.dev/sentio1/sentio/sentio-web:latest

echo "(3/4)Restarting pods..."
kubectl rollout restart deployment/sentio-web -n sentio

echo "(4/4)Waiting for rollout..."
kubectl rollout status deployment/sentio-web -n sentio

echo " Deployment Done! Access Sentio online: http://34.51.186.204"