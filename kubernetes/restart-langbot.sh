#!/usr/bin/env bash
docker buildx build --platform linux/amd64 -t registry-prod.deer-neon.ts.net/langbot:latest .. --push
kubectl --context sigsrv-prod -n langbot rollout restart statefulsets langbot-claude
kubectl --context sigsrv-prod -n langbot rollout restart statefulsets langbot-gpt-4o
