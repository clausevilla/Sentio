# Sentio ML Platform - Deployment Guide

## Overview

This guide will help you set up your development environment and deploy changes to our production server running on Google Kubernetes Engine (GKE). Our application is hosted at http://34.51.186.204 and uses PostgreSQL for the database and Google Cloud Storage for ML model files.

---

## Part 1: Setting Up Your Environment

Before you can deploy, you need to install some tools and authenticate with Google Cloud.

### 1.1 Install Required Tools

**Google Cloud SDK (gcloud)**

This is the command-line tool for interacting with Google Cloud services. Download and install it from https://cloud.google.com/sdk/docs/install. Follow the installer instructions for your operating system (Windows, macOS, or Linux).

**Docker**

We use Docker to build container images. Download and install Docker Desktop from https://www.docker.com/products/docker-desktop. After installation, make sure Docker is running (you should see the Docker icon in your system tray).

**DBeaver (Optional)**

DBeaver is a database GUI tool that lets you browse and query the production database visually. Download it from https://dbeaver.io/download/ if you want to inspect the database directly.

### 1.2 Authenticate with Google Cloud

Open your terminal and run the following commands one by one.

First, log in with your Google account. This will open a browser window where you can sign in:

```bash
gcloud auth login
```

Next, set our project as the default so you don't have to specify it every time:

```bash
gcloud config set project sentio1
```

Configure Docker to authenticate with our private container registry. This allows you to push and pull Docker images:

```bash
gcloud auth configure-docker europe-north2-docker.pkg.dev
```

Install kubectl, which is the command-line tool for managing Kubernetes:

```bash
gcloud components install kubectl
```

Finally, connect to our Kubernetes cluster. This downloads the cluster credentials and configures kubectl to communicate with it:

```bash
gcloud container clusters get-credentials sentio-cluster-1 --region europe-north2
```

### 1.3 Verify Your Setup

Run this command to check that everything is working:

```bash
kubectl get pods -n sentio
```

You should see output like this showing our running pods:

```
NAME                          READY   STATUS    RESTARTS   AGE
nginx-bd7b84bf6-hwc4t         1/1     Running   0          2h
sentio-web-5d4f8b7c9-abcde    2/2     Running   0          2h
sentio-web-5d4f8b7c9-fghij    2/2     Running   0          2h
```

If you see this, congratulations! You're ready to deploy.

---

## Part 2: Deploying Code Changes

When you've made changes to the code and want to deploy them to production, follow these steps.

### 2.1 Using the Deploy Script

The simplest way to deploy is using our automated script. From the project root directory, run:

```bash
./deploy.sh
```

This script does the following automatically:

1. Builds a new Docker image with your code changes
2. Pushes the image to Google Container Registry
3. Updates the Kubernetes deployment to use the new image
4. Waits for the deployment to complete

The script will show you the progress and let you know when the deployment is finished.

### 2.2 Restarting Without Code Changes

Sometimes you just need to restart the application (for example, after changing environment variables or to clear a stuck process). In this case, you don't need to rebuild the image:

```bash
kubectl rollout restart deployment/sentio-web -n sentio
```

This restarts all the Django application pods with the existing image.

---

## Part 3: Database Migrations

When you make changes to Django models (adding fields, creating new models, etc.), you need to create and apply database migrations. This is a multi-step process that involves both your local machine and the production server.

### 3.1 The Migration Workflow

Here's the complete workflow from making a model change to having it live in production.

**Step 1: Make your model changes locally**

Edit your model file. For example, if you're adding a new field to `ModelVersion`:

```python
# apps/ml_admin/models.py
class ModelVersion(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True)  # New field you're adding
    # ... rest of the model
```

**Step 2: Create the migration file**

Django needs to generate a migration file that describes your changes. Run this command locally:

```bash
python manage.py makemigrations
```

You'll see output confirming the migration was created:

```
Migrations for 'ml_admin':
  apps/ml_admin/migrations/0005_modelversion_description.py
    - Add field description to modelversion
```

**Step 3: Test the migration locally**

Before deploying, make sure the migration works on your local database:

```bash
python manage.py migrate
```

This applies the migration to your local SQLite database. If there are any errors, fix them before proceeding.

**Step 4: Commit the migration file**

Migration files must be committed to git so they get deployed with your code:

```bash
git add apps/ml_admin/migrations/0005_modelversion_description.py
git commit -m "Add description field to ModelVersion"
git push
```

**Step 5: Deploy to production**

Now deploy your code (which includes the new migration file):

```bash
./deploy.sh
```

**Step 6: Apply the migration to the production database**

After the deployment finishes, the new migration file exists on the server but hasn't been applied to the database yet. Run this command to apply it:

```bash
kubectl exec -n sentio deployment/sentio-web -c web -- python manage.py migrate
```

**Step 7: Verify the migration was applied**

Check that all migrations show as applied (marked with [X]):

```bash
kubectl exec -n sentio deployment/sentio-web -c web -- python manage.py showmigrations ml_admin
```

### 3.2 Transferring Data from Local to Production

If you have data in your local SQLite database that needs to be copied to production (for example, model records you created during development), use Django's dumpdata and loaddata commands.

**Export data from your local database:**

```bash
python manage.py dumpdata ml_admin.ModelVersion --indent 2 > data.json
```

This creates a JSON file containing all ModelVersion records.

**Copy the file to the production pod:**

```bash
kubectl cp data.json sentio/$(kubectl get pods -n sentio -l app=sentio-web -o jsonpath='{.items[0].metadata.name}'):/app/data.json -c web
```

**Import the data into production:**

```bash
kubectl exec -n sentio deployment/sentio-web -c web -- python manage.py loaddata /app/data.json
```

---

## Part 4: Viewing the Database with DBeaver

DBeaver lets you browse the production database visually, run queries, and export data. Since our Cloud SQL database isn't directly accessible from the internet, we need to create a secure tunnel first.

### 4.1 Start the Port Forward

Open a terminal and run this command. Keep this terminal open the entire time you're using DBeaver:

```bash
kubectl port-forward -n sentio deployment/sentio-web 5432:5432
```

You should see:

```
Forwarding from 127.0.0.1:5432 -> 5432
```

This creates a tunnel from your computer (localhost:5432) to the database.

### 4.2 Create a Connection in DBeaver

Open DBeaver and create a new connection:

1. Go to **Database** → **New Database Connection**
2. Select **PostgreSQL** from the list and click **Next**
3. Fill in the connection details:
   * **Host:** localhost
   * **Port:** 5432
   * **Database:** sentio_db
   * **Username:** sentio_user
   * **Password:** see Part 5.2 to retrieve from Secret Manager
4. Click **Test Connection** to make sure it works
5. Click **Finish** to save the connection

### 4.3 Browsing the Database

Once connected, you can find tables at: **sentio_db** → **Schemas** → **public** → **Tables**

Double-click any table to view its data. You can also open a SQL editor to run custom queries.

**Important:** Remember to keep the port-forward terminal running. If you close it, DBeaver will lose the connection.

---

## Part 5: Managing Secrets

All sensitive information (database passwords, API keys, Django secret key) is stored in Google Secret Manager, not in code or config files. This keeps our secrets secure and out of git.

### 5.1 Viewing Secrets in the Browser

The easiest way to view secrets is through the Google Cloud Console:

1. Go to https://console.cloud.google.com/security/secret-manager?project=sentio1
2. You'll see a list of all our secrets
3. Click on any secret name to see its details
4. Click **View secret value** to reveal the actual value

### 5.2 Viewing Secrets via Command Line

To list all available secrets:

```bash
gcloud secrets list --project=sentio1
```

To view a specific secret's value (for example, the database password):

```bash
gcloud secrets versions access latest --secret="sql-password" --project=sentio1
```

### 5.3 Adding a New Secret

If you need to add a new secret:

```bash
echo -n "your-secret-value" | gcloud secrets create YOUR_SECRET_NAME --data-file=- --project=sentio1
```

After adding a secret, you'll need to update the Kubernetes deployment to use it and restart the pods.

---

## Part 6: Uploading ML Models

Trained ML models are stored in Google Cloud Storage (GCS), not in the code repository. The bucket location is `gs://sentio-m_l-models/`.

### 6.1 Upload Models

To upload a single model:

```bash
gsutil cp path/to/your/model.pt gs://sentio-m_l-models/models/
```

To upload multiple models at once (the -m flag enables parallel uploads for speed):

```bash
gsutil -m cp path/to/models/*.pt gs://sentio-m_l-models/models/
```

### 6.2 Verify Upload

To see what models are in the bucket:

```bash
gsutil ls gs://sentio-m_l-models/models/
```

---

## Part 7: Common Commands Reference

### Viewing Logs

To see what's happening in the application, check the logs.

**Django application logs (live streaming):**

```bash
kubectl logs -n sentio deployment/sentio-web -c web -f
```

Press Ctrl+C to stop streaming.

**Django application logs (last 100 lines):**

```bash
kubectl logs -n sentio deployment/sentio-web -c web --tail=100
```

**Nginx logs:**

```bash
kubectl logs -n sentio deployment/nginx -f
```

### Checking Pod Status

**List all pods and their status:**

```bash
kubectl get pods -n sentio
```

**Get detailed information about a specific pod (useful for debugging startup issues):**

```bash
kubectl describe pod -n sentio <pod-name>
```

### Accessing the Server Directly

**Open a bash shell inside the Django container:**

```bash
kubectl exec -n sentio deployment/sentio-web -c web -it -- bash
```

**Open the Django shell for running Python code:**

```bash
kubectl exec -n sentio deployment/sentio-web -c web -it -- python manage.py shell
```

**Create a new admin user:**

```bash
kubectl exec -n sentio deployment/sentio-web -c web -it -- python manage.py createsuperuser
```

---

## Part 8: Troubleshooting

### Pods Won't Start

If pods are stuck in "Pending", "CrashLoopBackOff", or "ImagePullBackOff" status:

First, check the pod events for error messages:

```bash
kubectl describe pod -n sentio <pod-name>
```

Look at the "Events" section at the bottom for clues about what's wrong.

If the pod crashed, check the logs from the previous run:

```bash
kubectl logs -n sentio <pod-name> -c web --previous
```

### 502 Bad Gateway Error

This usually means the Django application isn't responding. Check the Django logs for errors:

```bash
kubectl logs -n sentio deployment/sentio-web -c web --tail=50
```

### Static Files Not Loading (404 errors on CSS/JS)

Run collectstatic to regenerate static files, then restart the pods:

```bash
kubectl exec -n sentio deployment/sentio-web -c web -- python manage.py collectstatic --noinput
kubectl rollout restart deployment/sentio-web -n sentio
```

### Database Connection Issues

Check if the Cloud SQL proxy container is running properly:

```bash
kubectl logs -n sentio deployment/sentio-web -c cloud-sql-proxy --tail=50
```

---

## Quick Reference URLs

**Application:** http://34.51.186.204

**Our own ML-Admin Panel:** http://34.51.186.204/management/ (username: admin, password: admin123)

**Google Secret Manager:** https://console.cloud.google.com/security/secret-manager?project=sentio1

**Google Cloud Storage:** https://console.cloud.google.com/storage/browser/sentio-m_l-models?project=sentio1
