# SpecHO Deployment Guide

**Version:** 1.0  
**Prerequisites:** Tier 1 complete and validated  
**Target Audience:** DevOps, deployment engineers, system administrators

---

## Overview

This guide covers deploying SpecHO as a production service after Tier 1 validation is complete. The deployment strategy follows the same additive philosophy as development: start with a minimal working deployment, measure real-world performance, then enhance based on actual needs.

**Do not deploy until:**
- All 32 Tier 1 tasks are complete
- Integration tests pass consistently
- Baseline corpus is processed and validated
- False positive/negative rates are measured on 50+ real documents
- Performance benchmarks are documented

---

## Deployment Tiers

### Tier 1 Deployment: Local/Internal Testing
**When:** Immediately after Tier 1 development complete  
**Purpose:** Validate system with real users in controlled environment  
**Deployment method:** Docker Compose on single server  
**Users:** Internal team only (5-10 users)

### Tier 2 Deployment: Production Beta
**When:** After 2+ weeks of Tier 1 deployment data collection  
**Purpose:** Serve external users with monitored performance  
**Deployment method:** Kubernetes with load balancing  
**Users:** Beta testers and early adopters (100-1000 users)

### Tier 3 Deployment: Full Production
**When:** After Tier 2 shows < 5% false positive/negative rate  
**Purpose:** High-availability production system  
**Deployment method:** Multi-region Kubernetes with auto-scaling  
**Users:** General public (unlimited)

---

## Architecture Options

### Option A: Web API Service
Best for integration with existing applications.

**Components:**
- FastAPI backend exposing REST endpoints
- Redis for request queuing and caching
- PostgreSQL for storing analysis results
- Nginx reverse proxy

**Endpoints:**
```
POST /analyze
  - Body: {"text": "...", "options": {...}}
  - Returns: DocumentAnalysis JSON

GET /analysis/{id}
  - Returns: Previously computed analysis

POST /batch
  - Body: {"texts": [...], "options": {...}}
  - Returns: Job ID for async processing

GET /batch/{job_id}
  - Returns: Batch analysis status and results
```

### Option B: Web Interface
Best for direct user interaction and demonstrations.

**Components:**
- React/Vue frontend for text input and result visualization
- FastAPI backend (same as Option A)
- WebSocket support for real-time progress updates
- File upload support for batch processing

**Features:**
- Text paste or file upload (txt, docx, pdf)
- Real-time analysis progress bar
- Interactive result visualization showing clause pairs and echo scores
- Downloadable reports (JSON, PDF)
- Analysis history for logged-in users

### Option C: CLI Service
Best for automated workflows and batch processing.

**Components:**
- CLI tool (already implemented in Tier 1)
- Optional job scheduler (cron, Airflow) for batch processing
- Shared file system or S3 for input/output

**Use cases:**
- Batch processing of document libraries
- Integration with CI/CD pipelines for content validation
- Automated monitoring of content streams

---

## Docker Setup (Tier 1 Deployment)

### Directory Structure
```
SpecHO/
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── nginx.conf
│   └── .dockerignore
├── [existing SpecHO files]
└── [existing data/, tests/, scripts/]
```

### Dockerfile
```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy application code
COPY SpecHO/ ./SpecHO/
COPY scripts/ ./scripts/
COPY data/baseline/ ./data/baseline/

# Create necessary directories
RUN mkdir -p data/models data/corpus logs

# Expose port for web service (if using web interface)
EXPOSE 8000

# Default command (can be overridden in docker-compose)
CMD ["python", "-m", "uvicorn", "scripts.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### docker-compose.yml (Basic Setup)
```yaml
version: '3.8'

services:
  specho-api:
    build:
      context: .
      dockerfile: docker/Dockerfile
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    environment:
      - SPECHO_CONFIG=simple
      - LOG_LEVEL=INFO
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### docker-compose.yml (Full Stack with Redis & PostgreSQL)
```yaml
version: '3.8'

services:
  specho-api:
    build:
      context: .
      dockerfile: docker/Dockerfile
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    environment:
      - SPECHO_CONFIG=simple
      - LOG_LEVEL=INFO
      - REDIS_URL=redis://redis:6379/0
      - DATABASE_URL=postgresql://specho:password@postgres:5432/specho
    depends_on:
      - redis
      - postgres
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    restart: unless-stopped

  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_USER=specho
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=specho
    ports:
      - "5432:5432"
    volumes:
      - postgres-data:/var/lib/postgresql/data
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./docker/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./docker/ssl:/etc/nginx/ssl:ro
    depends_on:
      - specho-api
    restart: unless-stopped

volumes:
  redis-data:
  postgres-data:
```

### Building and Running
```bash
# Build the Docker image
docker-compose build

# Run in foreground (for testing)
docker-compose up

# Run in background (for production)
docker-compose up -d

# View logs
docker-compose logs -f specho-api

# Stop services
docker-compose down

# Stop and remove volumes (caution: deletes data)
docker-compose down -v
```

---

## Kubernetes Deployment (Tier 2)

### Prerequisites
- Kubernetes cluster (1.25+)
- kubectl configured
- Container registry (Docker Hub, GCR, ECR)
- Persistent volume provisioner

### Deployment Strategy
1. Build and push Docker image to registry
2. Create Kubernetes namespace
3. Deploy ConfigMap for configuration
4. Deploy Secret for sensitive data
5. Deploy StatefulSet for PostgreSQL
6. Deploy Deployment for Redis
7. Deploy Deployment for SpecHO API (with HPA)
8. Deploy Service and Ingress for external access

### Example Kubernetes Manifests

**namespace.yaml**
```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: specho
```

**configmap.yaml**
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: specho-config
  namespace: specho
data:
  SPECHO_CONFIG: "simple"
  LOG_LEVEL: "INFO"
  REDIS_URL: "redis://redis-service:6379/0"
```

**deployment.yaml**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: specho-api
  namespace: specho
spec:
  replicas: 3
  selector:
    matchLabels:
      app: specho-api
  template:
    metadata:
      labels:
        app: specho-api
    spec:
      containers:
      - name: specho
        image: your-registry/specho:latest
        ports:
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: specho-config
        - secretRef:
            name: specho-secrets
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
```

**hpa.yaml** (Horizontal Pod Autoscaler)
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: specho-hpa
  namespace: specho
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: specho-api
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

**service.yaml**
```yaml
apiVersion: v1
kind: Service
metadata:
  name: specho-service
  namespace: specho
spec:
  selector:
    app: specho-api
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP
```

**ingress.yaml**
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: specho-ingress
  namespace: specho
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/rate-limit: "100"
spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - specho.yourdomain.com
    secretName: specho-tls
  rules:
  - host: specho.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: specho-service
            port:
              number: 80
```

### Deployment Commands
```bash
# Apply all manifests
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secrets.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml
kubectl apply -f k8s/ingress.yaml

# Check deployment status
kubectl get pods -n specho
kubectl get svc -n specho
kubectl get ingress -n specho

# View logs
kubectl logs -f deployment/specho-api -n specho

# Scale manually (if needed)
kubectl scale deployment specho-api --replicas=5 -n specho
```

---

## Monitoring and Observability

### Metrics to Track
**Performance metrics:**
- Request latency (p50, p95, p99)
- Throughput (requests per second)
- Error rate
- CPU and memory usage per component
- Queue depth (if using async processing)

**Business metrics:**
- Total documents analyzed
- Average document length
- Distribution of confidence scores
- False positive/negative rates (requires manual validation)
- User feedback ratings

### Logging Strategy
**Tier 1 logging:**
- Application logs to stdout/stderr
- Docker captures to logs volume
- Retain 7 days of logs

**Tier 2 logging:**
- Centralized logging with ELK Stack or Loki
- Structured JSON logs
- Retain 30 days of logs
- Indexed by request_id for tracing

**Log levels:**
- DEBUG: Detailed component behavior (development only)
- INFO: Request/response summary, component initialization
- WARNING: Degraded performance, fallback behavior
- ERROR: Failed requests, component failures
- CRITICAL: System-wide failures

### Recommended Tools
- **Metrics:** Prometheus + Grafana
- **Logging:** ELK Stack (Elasticsearch, Logstash, Kibana) or Grafana Loki
- **Tracing:** Jaeger or Zipkin (Tier 3)
- **Alerting:** Prometheus Alertmanager or PagerDuty
- **Uptime monitoring:** UptimeRobot, Pingdom, or internal healthchecks

---

## Security Considerations

### API Security
- Rate limiting (100 requests/hour per IP in Tier 1)
- API key authentication for production
- Input validation (max text length, file size limits)
- HTTPS only (TLS 1.3)
- CORS configuration for web interface

### Data Privacy
- No persistent storage of analyzed text by default
- Optional anonymization of stored results
- GDPR-compliant data retention policies
- User consent for optional result storage

### Infrastructure Security
- Network policies to isolate components
- Secrets management (Kubernetes Secrets, HashiCorp Vault)
- Regular security updates for base images
- Minimal container privileges (non-root user)

---

## Backup and Recovery

### Data to Backup
- Baseline corpus statistics (`data/baseline/`)
- Configuration files
- Trained models (if any in Tier 2+)
- Database (if storing analysis results)
- User data (if applicable)

### Backup Strategy
**Tier 1:**
- Daily backup of data/ directory to external storage
- Retention: 7 days

**Tier 2:**
- Automated database backups every 6 hours
- Baseline corpus backup weekly
- Retention: 30 days
- Cross-region replication

**Recovery Testing:**
- Test restore procedure monthly
- Document recovery time objective (RTO): 4 hours
- Document recovery point objective (RPO): 6 hours

---

## Deployment Checklist

### Pre-Deployment
- [ ] All Tier 1 tasks complete and tested
- [ ] Integration tests passing (>80% coverage)
- [ ] Baseline corpus processed and validated
- [ ] False positive/negative rate measured
- [ ] Performance benchmarks documented
- [ ] Docker image built and tested locally
- [ ] Environment variables configured
- [ ] SSL certificates obtained
- [ ] Monitoring dashboards configured
- [ ] Backup strategy implemented
- [ ] Incident response plan documented

### Initial Deployment
- [ ] Deploy to staging environment first
- [ ] Run smoke tests against staging
- [ ] Load test with realistic traffic patterns
- [ ] Verify monitoring and logging
- [ ] Test backup and restore procedures
- [ ] Deploy to production
- [ ] Monitor for 24 hours with increased alerting
- [ ] Gather initial performance data

### Post-Deployment
- [ ] Document any deployment issues encountered
- [ ] Update runbooks with real-world findings
- [ ] Schedule first maintenance window
- [ ] Begin collecting user feedback
- [ ] Review metrics weekly for first month
- [ ] Identify Tier 2 enhancement priorities based on real usage

---

## Rollback Procedure

If deployment fails or critical issues are discovered:

1. **Immediate rollback** (< 5 minutes):
   ```bash
   # Docker Compose
   docker-compose down
   git checkout previous-stable-tag
   docker-compose up -d
   
   # Kubernetes
   kubectl rollout undo deployment/specho-api -n specho
   ```

2. **Verify rollback**:
   - Check application logs
   - Test critical endpoints
   - Verify monitoring shows normal behavior

3. **Post-incident**:
   - Document what went wrong
   - Create fix in development environment
   - Test fix thoroughly before redeploying
   - Update deployment checklist with new checks

---

## Scaling Guidelines

### When to Scale Up
- CPU usage consistently > 70%
- Memory usage consistently > 80%
- Request latency p95 > 2 seconds
- Queue depth growing over time
- Error rate > 1%

### When to Scale Down
- CPU usage consistently < 30% for 1 week
- Cost optimization identified
- Traffic patterns changed (post-analysis)

### Vertical vs Horizontal Scaling
**Vertical (increase resources per pod):**
- When single request requires more memory/CPU
- Limited by node size in Kubernetes
- Simpler but less fault-tolerant

**Horizontal (increase number of pods):**
- When request volume increases
- Better fault tolerance and availability
- Requires stateless design (already true for SpecHO)
- Preferred approach for Tier 2+

---

## Maintenance

### Regular Maintenance Tasks
**Daily:**
- Review error logs
- Check disk space usage
- Verify backup completion

**Weekly:**
- Review performance metrics
- Check for dependency updates
- Review user feedback

**Monthly:**
- Update base Docker images for security patches
- Review and tune autoscaling policies
- Test disaster recovery procedures
- Review and optimize resource allocation

### Upgrade Process
1. Test upgrade in development environment
2. Deploy to staging
3. Run full integration test suite
4. Deploy to production during low-traffic window
5. Monitor closely for 24 hours
6. Document any issues or configuration changes

---

## Troubleshooting

### Common Issues

**High latency:**
- Check CPU/memory usage
- Review recent code changes
- Check database query performance
- Verify network connectivity
- Review baseline corpus loading time

**High error rate:**
- Check application logs for stack traces
- Verify all dependencies are available (spaCy model, baseline data)
- Check input validation (are invalid texts being sent?)
- Review recent configuration changes

**Out of memory:**
- Check document sizes being processed
- Review memory limits in deployment config
- Consider implementing document size limits
- Check for memory leaks (gradual increase over time)

**Deployment failures:**
- Verify Docker image builds successfully
- Check environment variables are set correctly
- Verify all required volumes are mounted
- Review pod events: `kubectl describe pod <pod-name> -n specho`

---

## Support and Resources

**Documentation:**
- This deployment guide
- [TASKS.md](TASKS.md) for component architecture
- [SPECS.md](SPECS.md) for detailed specifications
- [PHILOSOPHY.md](PHILOSOPHY.md) for design decisions

**Monitoring dashboards:**
- Application metrics: `http://grafana.yourdomain.com/specho`
- System logs: `http://kibana.yourdomain.com`

**Getting help:**
- File issues in project repository
- Check application logs first
- Include request_id when reporting issues
- Provide steps to reproduce

---

## Next Steps After Deployment

Once Tier 1 deployment is stable (2+ weeks of operation):

1. **Analyze real-world usage patterns**:
   - What document types are most common?
   - What are actual latency requirements?
   - Where are false positives/negatives occurring?

2. **Identify Tier 2 priorities** based on data:
   - Performance bottlenecks that impact users
   - Accuracy improvements with highest ROI
   - Features requested by multiple users

3. **Plan Tier 2 enhancements**:
   - Review SPECS.md for Tier 2 options
   - Prioritize based on measured impact
   - Update deployment architecture as needed

Remember: The goal is not to build everything possible, but to build what real usage proves is necessary.

---

**Last Updated:** [Current Date]  
**Maintained By:** [Team/Individual]  
**Review Schedule:** Monthly during Tier 1, quarterly after stable
