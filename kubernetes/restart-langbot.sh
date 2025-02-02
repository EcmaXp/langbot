#!/usr/bin/env bash
docker buildx build --platform linux/amd64 -t registry-t1.deer-neon.ts.net/langbot:latest .. --push
kubectl --context sigsrv-prod -n langbot rollout restart deployment langbot-claude
kubectl --context sigsrv-prod -n langbot rollout restart deployment langbot-openai-gpt-4o
kubectl --context sigsrv-prod -n langbot rollout restart deployment langbot-openai-o1
kubectl --context sigsrv-prod -n langbot rollout restart deployment langbot-openai-o3-mini
