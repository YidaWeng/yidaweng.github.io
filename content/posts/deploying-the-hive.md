---
title: "Deploying The Hive: Taking a Multi-Agent Engineering System to Production on GKE"
date: 2026-07-18
description: "A complete walkthrough of containerizing, provisioning cloud infrastructure on GCloud, and deploying a multi-agent AI engineering platform to GKE Autopilot using Terraform, Docker, and Cloud Build"
tags: ["kubernetes", "gke", "terraform", "docker", "cloud-build", "devops", "infrastructure", "gcloud"]
categories: ["DevOps"]
toc: true
cover:
  image: "images/hive-gke-architecture.png"
  alt: "GKE deployment architecture diagram"
  caption: "The Hive deployment architecture on Google Kubernetes Engine"
---

## The Problem

Multi-agent AI systems are notoriously hard to productionize. They depend on databases (PostgreSQL, Redis), vector stores (ChromaDB), sandboxed code execution (Docker-in-Docker), and increasingly support service mesh sidecars — all while needing observability, autoscaling, and CI/CD. Running this stack locally with `docker-compose` gets you to prototype, but taking it to production requires infrastructure-as-code, container orchestration, and a deployment pipeline.

This post walks through the entire journey: containerizing [The Hive](https://github.com/YidaWeng/Swarm) — a LangGraph-powered multi-agent engineering system — and deploying it to Google Kubernetes Engine (GKE) Autopilot using Terraform, Docker, and Cloud Build. Every file discussed here is checked into the repository for reproducibility.

---

## Architecture Overview

The Hive has five runtime dependencies that shaped our deployment strategy:

| Component | Role | Deployment Strategy |
|-----------|------|--------------------|
| **FastAPI gateway** | REST API, workflow orchestration | GKE Deployment (container) |
| **PostgreSQL 15** | Persistent workflow state, project context | Cloud SQL (managed) |
| **Redis 7** | Session state, short-term memory | Memorystore (managed) |
| **ChromaDB** | Vector embeddings, cognitive memory | GKE Deployment (self-hosted) |
| **Docker sandbox** | Isolated code execution | GKE pod with Docker socket / DinD sidecar |

The decision to use GKE Autopilot over Cloud Run was driven by two constraints: the sandbox requires access to a Docker daemon (DinD sidecar or host socket), and future Go sidecars (proxy, telemetry, token metering) will need to run as mesh sidecars alongside the main container. Autopilot gives us Kubernetes orchestration without managing nodes.

For managed services, we use **Cloud SQL** over containerized Postgres (backups, maintenance, high availability built in) and **Memorystore** over containerized Redis (same reasoning). ChromaDB runs on-cluster because there's no managed GCloud equivalent — it gets a PersistentVolumeClaim backed by standard-rwo SSD.

---

## Dockerizing for Production

The original Dockerfile had a modest goal: build and run locally. For production we needed three changes.

**Before — the builder stage installed dev dependencies and required an editable install:**

```dockerfile
COPY pyproject.toml .
RUN pip install --no-cache-dir -e .[dev]
```

This installs pytest, ruff, mypy, and other dev tools into the production image. Worse, `pip install -e .` requires the package source directories to exist, but only `pyproject.toml` was copied — the builder stage would fail if we had strict build checks.

**After — production dependencies from `requirements.txt`:**

```dockerfile
FROM python:3.11-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
```

This installs only runtime dependencies (FastAPI, sqlalchemy, chromadb, redis-py, etc.) and drops the editable install dance. The final stage copies site-packages from the builder, then copies the application source separately.

We also added two runtime necessities:

```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    docker.io \
    postgresql-client \
    curl \
    && rm -rf /var/lib/apt/lists/*

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
```

- **curl** is required for the HEALTHCHECK instruction
- **docker.io** keeps the sandbox runner functional (requires host Docker socket mount at runtime)
- **postgresql-client** is available for debugging but not strictly required by the app

The healthcheck endpoint already exists at `GET /health` in the FastAPI app and returns component status for the orchestrator, database, Redis, and vector store.

Finally, a `.dockerignore` was added to keep the build context lean:

```
.git/
__pycache__/
.env
tests/
*.md
htmlcov/
```

---

## Infrastructure as Code with Terraform

We manage all GCloud resources through Terraform, organized into five files:

```
infra/terraform/
├── versions.tf    # Provider version constraints
├── provider.tf    # GCloud + Kubernetes provider config
├── variables.tf   # All input variables
├── main.tf        # Resources
└── outputs.tf     # Connection info + helper commands
```

### VPC and Networking

The VPC uses custom subnet mode with secondary CIDR ranges for GKE pods and services:

```hcl
module "vpc" {
  source       = "terraform-google-modules/network/google"
  network_name = "hive-dev-vpc"
  routing_mode = "REGIONAL"

  subnets = [
    {
      subnet_name           = "hive-dev-subnet"
      subnet_ip             = "10.0.0.0/20"
      subnet_region         = var.region
      subnet_private_access = true
    }
  ]

  secondary_ranges = {
    "hive-dev-subnet" = [
      { range_name = "pods", ip_cidr_range = "10.1.0.0/16" },
      { range_name = "services", ip_cidr_range = "10.2.0.0/20" },
    ]
  }
}
```

### Managed Services

Cloud SQL and Memorystore are provisioned on the private network so the GKE cluster can reach them over internal IPs:

```hcl
resource "google_sql_database_instance" "postgres" {
  database_version = "POSTGRES_15"
  settings {
    tier              = "db-custom-1-3840"
    disk_type         = "PD_SSD"
    disk_autoresize   = true
    availability_type = "ZONAL"
    ip_configuration {
      ipv4_enabled    = false
      private_network = module.vpc.network_id
    }
  }
}

resource "google_redis_instance" "redis" {
  tier           = "basic"
  memory_size_gb = 1
  connect_mode   = "PRIVATE_SERVICE_ACCESS"
  authorized_network = module.vpc.network_id
}
```

### GKE Autopilot

The GKE module creates a regional Autopilot cluster with Workload Identity enabled for secure access to GCloud APIs:

```hcl
module "gke" {
  source     = "terraform-google-modules/kubernetes-engine/google//modules/beta-autopilot-public-cluster"
  name       = "hive-dev-cluster"
  regional   = true
  region     = var.region
  network    = module.vpc.network_name
  subnetwork = module.vpc.subnets_names[0]
  ip_range_pods     = "pods"
  ip_range_services = "services"
  release_channel   = "REGULAR"
}
```

The Kubernetes provider is configured to connect to the newly-created cluster automatically:

```hcl
provider "kubernetes" {
  host                   = "https://${module.gke.endpoint}"
  token                  = data.google_client_config.default.access_token
  cluster_ca_certificate = base64decode(module.gke.ca_certificate)
}
```

This lets us apply Kubernetes resources directly from the same Terraform run. The ConfigMap and Secret are created alongside the infrastructure:

```hcl
resource "kubernetes_config_map" "hive_config" {
  metadata { name = "hive-config"; namespace = "hive" }
  data = {
    POSTGRES_HOST  = google_sql_database_instance.postgres.private_ip_address
    REDIS_HOST     = google_redis_instance.redis.host
    CHROMA_HOST    = "chroma-service.hive.svc.cluster.local"
    # ... other env vars
  }
}
```

### Applying the Infrastructure

```bash
cd infra/terraform
terraform init
terraform apply -var="project_id=my-gcp-project" -var="openai_api_key=sk-..."
```

Terraform provisions everything — VPC, Cloud SQL, Memorystore, Artifact Registry, GKE cluster, and the initial Kubernetes namespace with ConfigMap. The `outputs.tf` file prints connection details and the `gcloud` command to configure `kubectl`.

---

## Deploying the Stack to Kubernetes

### ChromaDB

ChromaDB runs as a standard Deployment with a PersistentVolumeClaim. We pin the version to `0.5.0` and configure token-based authentication:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: chroma
  namespace: hive
spec:
  replicas: 1
  selector:
    matchLabels:
      app: chroma
  template:
    spec:
      containers:
        - name: chroma
          image: chromadb/chroma:0.5.0
          ports:
            - containerPort: 8000
          env:
            - name: CHROMA_SERVER_AUTH_CREDENTIALS_PROVIDER
              value: "chromadb.auth.token.TokenAuthServerProvider"
            - name: CHROMA_SERVER_AUTH_CREDENTIALS
              value: "test-token"
            - name: IS_PERSISTENT
              value: "true"
          volumeMounts:
            - name: chroma-data
              mountPath: /chroma/chroma
          resources:
            requests: { cpu: "500m", memory: "512Mi" }
            limits:   { cpu: "1", memory: "1Gi" }
      volumes:
        - name: chroma-data
          persistentVolumeClaim:
            claimName: chroma-pvc
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: chroma-pvc
  namespace: hive
spec:
  accessModes: ["ReadWriteOnce"]
  resources: { requests: { storage: 10Gi } }
  storageClassName: standard-rwo
```

A headless ClusterIP service provides DNS-based discovery:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: chroma-service
  namespace: hive
spec:
  selector:
    app: chroma
  ports:
    - port: 8000
      targetPort: 8000
  type: ClusterIP
```

The `ChromaStore` implementation auto-detects the deployment mode at runtime: if `CHROMA_HOST` is set to a non-localhost address (e.g. `chroma-service.hive.svc.cluster.local` on K8s), it switches to `chromadb.HttpClient` and communicates over the network. When `CHROMA_HOST` is `localhost` (local development), it uses `chromadb.PersistentClient` with a local directory. This enables zero-config switching between `docker-compose` and GKE deployments.

### The Hive Application

The main deployment reads configuration from the Terraform-managed ConfigMap and Secret:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hive
  namespace: hive
spec:
  replicas: 1
  selector:
    matchLabels:
      app: hive
  template:
    metadata:
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
    spec:
      containers:
        - name: hive
          image: us-central1-docker.pkg.dev/the-hive-project/hive-dev-images/hive:latest
          ports:
            - containerPort: 8000
          envFrom:
            - configMapRef: { name: hive-config }
            - secretRef:    { name: hive-secrets }
          resources:
            requests: { cpu: "500m", memory: "512Mi" }
            limits:   { cpu: "2", memory: "2Gi" }
          livenessProbe:
            httpGet: { path: /health, port: 8000 }
            initialDelaySeconds: 30
          readinessProbe:
            httpGet: { path: /health, port: 8000 }
            initialDelaySeconds: 10
```

Key details:
- **Probes**: Both liveness and readiness point at `/health`. The 30-second initial delay gives Postgres table creation and Redis connection time during startup.
- **Prometheus annotations**: Enable metric scraping without additional exporters.
- **Horizontal autoscaling**: An HPA scales from 1 to 5 replicas at 70% CPU or 80% memory utilization.

### Service Mesh Considerations

The Docker sandbox runner (`infra/sandbox/docker_runner.py`) uses the `docker-py` library and calls `docker.from_env()`, which connects to the Docker socket. On GKE Autopilot, Docker socket access requires either:

1. **A DinD sidecar** — runs a Docker daemon in a privileged container sharing `/var/run/docker.sock` with the main container
2. **Host socket mount** — not supported on Autopilot (no hostPath volumes)

The DinD approach works on Autopilot but goes against the security model. For production, the sandbox should be refactored to call an external sandbox API (e.g., E2B or a separate sandbox service on GKE) instead of spawning containers directly.

The Go sidecars (`proxy/`, `telemetry/`, `token_meter/` in `infra/go_sidecars/`) are currently stubs — only `go.mod` with gRPC dependencies exists. When implemented, these will run as init containers or sidecar containers in the same pod, forwarding traces and enforcing rate limits via gRPC.

### Ingress

A GCE Ingress provides external HTTPS access to the API:

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: hive-ingress
  namespace: hive
  annotations:
    kubernetes.io/ingress.class: "gce"
    kubernetes.io/ingress.global-static-ip-name: "hive-ingress-ip"
    networking.gke.io/managed-certificates: "hive-cert"
spec:
  defaultBackend:
    service:
      name: hive-service
      port:
        number: 80
```

This provisions a Cloud Load Balancer with a static IP and an auto-renewed managed TLS certificate. The backend `hive-service` is a ClusterIP service on port 80 forwarding to the FastAPI container on port 8000.

---

## CI/CD with Cloud Build

The CI/CD pipeline is defined in a single `cloudbuild.yaml` at the repository root:

```yaml
steps:
  - id: "build-image"
    name: "gcr.io/cloud-builders/docker"
    args:
      - "build"
      - "-t"
      - "$_REGION-docker.pkg.dev/$PROJECT_ID/$_REPOSITORY/hive:$SHORT_SHA"
      - "-t"
      - "$_REGION-docker.pkg.dev/$PROJECT_ID/$_REPOSITORY/hive:latest"
      - "."

  - id: "push-image"
    name: "gcr.io/cloud-builders/docker"
    args:
      - "push"
      - "$_REGION-docker.pkg.dev/$PROJECT_ID/$_REPOSITORY/hive:$SHORT_SHA"

  - id: "update-manifest"
    name: "bash"
    args:
      - "-c"
      - |
        sed -i \
          "s|image: .*|image: $_REGION-docker.pkg.dev/$PROJECT_ID/$_REPOSITORY/hive:$SHORT_SHA|" \
          infra/kubernetes/hive-deployment.yaml

  - id: "deploy-to-gke"
    name: "gcr.io/cloud-builders/kubectl"
    args:
      - "apply"
      - "-f"
      - "infra/kubernetes/"
    env:
      - "CLOUDSDK_COMPUTE_REGION=$_REGION"
      - "CLOUDSDK_CONTAINER_CLUSTER=$_CLUSTER_NAME"

substitutions:
  _REGION: us-central1
  _REPOSITORY: hive-dev-images
  _CLUSTER_NAME: hive-dev-cluster
```

The pipeline runs four steps:

1. **Build** the Docker image, tagged with both the commit SHA and `latest`
2. **Push** to Artifact Registry
3. **Update** the deployment manifest with the new image tag
4. **Deploy** by applying the entire `infra/kubernetes/` directory

Cloud Build's `gcr.io/cloud-builders/kubectl` image authenticates to the GKE cluster automatically via the Cloud Build service account's `container.deploy` permission. The `sed` substitution pins the exact image SHA in the deployment manifest, enabling rollbacks by re-applying a previous manifest.

To trigger the pipeline:

```bash
gcloud builds submit --config=cloudbuild.yaml \
  --substitutions=_REGION=us-central1,_REPOSITORY=hive-dev-images,_CLUSTER_NAME=hive-dev-cluster
```

Or connect the repository to Cloud Build triggers for automatic deployment on every push to `main`.

---

## Deploying End-to-End

Here is the full workflow from scratch:

```bash
# 1. Authenticate
gcloud auth login
gcloud config set project the-hive-project

# 2. Provision infrastructure (first time only)
cd infra/terraform
terraform init
terraform apply -var="project_id=the-hive-project" \
  -var="openai_api_key=sk-..." \
  -var="anthropic_api_key=sk-ant-..." \
  -var="langchain_api_key=lsv2_..."

# 3. Connect to the new cluster
gcloud container clusters get-credentials hive-dev-cluster \
  --region us-central1 --project the-hive-project

# 4. Build and deploy
export POSTGRES_PASSWORD=$(terraform output -raw postgres_password)
make deploy

# 5. Verify
kubectl get pods -n hive
kubectl get svc -n hive
kubectl get ingress -n hive

# 6. Health check (via the load balancer IP)
curl -f http://<INGRESS_IP>/health
```

The `Makefile` wraps common workflows:

```makefile
build       # docker build
push        # docker build + push to Artifact Registry
k8s-apply   # kubectl apply -f infra/kubernetes/
deploy      # push + k8s-apply (full deploy)
tf-init     # terraform init
terraform-apply  # terraform apply
```

---

## What's Next

This deployment is production-ready for the current architecture, but several improvements are planned:

- **Go sidecar implementation**: The `proxy/`, `telemetry/`, and `token_meter/` sidecars exist as stubs with gRPC dependencies in `go.mod`. Implementing these will add request shaping, trace forwarding, and cost governance at the mesh layer.
- ~~**Chroma HTTP client**: Switching from `PersistentClient` to `HttpClient` will enable horizontal scaling~~ — **Done.** The `ChromaStore` now auto-detects HTTP vs Persistent mode from `CHROMA_HOST`.
- **External sandbox API**: Replacing the DinD sidecar with E2B or a dedicated sandbox microservice removes the privileged container requirement.
- **Observability stack**: Deploying Prometheus + Grafana (dashboards already defined in `observability/dashboards/`) for token burn, cost per run, and agent latency tracking, plus Jaeger for distributed tracing across the orchestrator and sidecars.
- **Terraform Cloud**: Migrating Terraform state to Terraform Cloud for team collaboration and run approvals.
- **Terraform test framework**: Adding `terraform test` assertions that validate the infrastructure against compliance rules before deployment.

All infrastructure and deployment code is available at [github.com/YidaWeng/Swarm](https://github.com/YidaWeng/Swarm) under `infra/terraform/` and `infra/kubernetes/`.

---

## References

- [Google Kubernetes Engine (GKE) Autopilot](https://cloud.google.com/kubernetes-engine/docs/concepts/autopilot-overview)
- [Terraform Google Provider](https://registry.terraform.io/providers/hashicorp/google/latest)
- [Cloud Build — CI/CD for GKE](https://cloud.google.com/build/docs/deploying-builds/deploy-gke)
- [ChromaDB — vector database](https://www.trychroma.com/)
- [The Hive — Multi-Agent Engineering System](https://github.com/YidaWeng/Swarm)
