---
description: Deploy application to production environment
allowed-tools: Bash(aws:*), Bash(kubectl:*), Bash(docker:*)
category: devops
depends-on: [code-review]
mcp-servers: [aws-mcp]
prerequisite-for: [monitoring-setup]
---

Deploy the application to the production environment.

Before deploying, ensure [[code-review]] has been completed so changes are verified.
Run pre-deployment checks including [[integration-tests]] to catch regressions.

## Build Phase

Build containers and push to registry. When using [[container-best-practices]]
the build process handles layer caching and multi-stage optimization.

## Rollout

Update Kubernetes manifests and verify the deployment is healthy.
Monitor logs for errors during rollout using [[monitoring-setup]] dashboards.
If issues arise, follow the [[rollback-procedure]] to restore the previous version.
