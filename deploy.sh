#!/bin/bash
# Author: Lian Shi
# Deploy script for Sentio ML Platform

set -e  # Exit on any error

echo "(1/4) Building image..."
docker build --platform linux/amd64 -t europe-north2-docker.pkg.dev/sentio1/sentio/sentio-web:latest . || { echo "❌ Build failed"; exit 1; }

echo "(2/4) Pushing image..."
docker push europe-north2-docker.pkg.dev/sentio1/sentio/sentio-web:latest || { echo "❌ Push failed"; exit 1; }

echo "(3/4) Restarting pods..."
kubectl rollout restart deployment/sentio-web -n sentio

echo "(4/4) Waiting for rollout..."
kubectl rollout status deployment/sentio-web -n sentio

echo ""
echo "Deployment Done!"
echo "Website: http://34.51.186.204"
echo "ML Admin:   http://34.51.186.204/management/"