#!/bin/bash
# Author: Lian Shi
# Deploy script for Sentio ML Platform

set -e  # Exit on any error

# ============================================
# IMAGE VERSIONING
# ============================================
REGISTRY="europe-north2-docker.pkg.dev/sentio1/sentio"
IMAGE_NAME="sentio-web"

# Use git commit SHA (short) as version tag
GIT_SHA=$(git rev-parse --short HEAD)
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

# Image tags
VERSION_TAG="${GIT_SHA}"
FULL_IMAGE="${REGISTRY}/${IMAGE_NAME}:${VERSION_TAG}"
LATEST_IMAGE="${REGISTRY}/${IMAGE_NAME}:latest"

echo "============================================"
echo "Deploying Sentio ML Platform"
echo "============================================"
echo "Git SHA:    ${GIT_SHA}"
echo "Image:      ${FULL_IMAGE}"
echo "============================================"
echo ""

# Check for uncommitted changes
if [ -n "$(git status --porcelain)" ]; then
    echo "⚠️  Warning: You have uncommitted changes!"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "(1/6) Building image..."
docker build --platform linux/amd64 \
    -t "${FULL_IMAGE}" \
    -t "${LATEST_IMAGE}" \
    . || { echo "❌ Build failed"; exit 1; }

echo "(2/6) Pushing versioned image (${VERSION_TAG})..."
docker push "${FULL_IMAGE}" || { echo "❌ Push failed"; exit 1; }

echo "(3/6) Pushing latest tag..."
docker push "${LATEST_IMAGE}" || { echo "❌ Push failed"; exit 1; }

echo "(4/6) Applying k8s manifests..."
kubectl apply -f k8s/web-deployment.yaml

echo "(5/6) Updating deployment image..."
kubectl set image deployment/sentio-web \
    -n sentio \
    web="${FULL_IMAGE}"

echo "(6/6) Waiting for rollout..."
kubectl rollout status deployment/sentio-web -n sentio

echo ""
echo "============================================"
echo "✅ Deployment Complete!"
echo "============================================"
echo "Version:    ${VERSION_TAG}"
echo "Image:      ${FULL_IMAGE}"
echo "Website:    http://34.51.186.204"
echo "ML Admin:   http://34.51.186.204/management/"
echo ""
echo "Rollback command:"
echo "  kubectl rollout undo deployment/sentio-web -n sentio"
echo "============================================"