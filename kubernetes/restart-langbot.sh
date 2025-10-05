#!/usr/bin/env bash
docker buildx build --platform linux/amd64 -t registry-t1.deer-neon.ts.net/langbot:latest .. --push
kubectl --context sigsrv-prod -n langbot rollout restart deployment langbot-claude-sonnet
kubectl --context sigsrv-prod -n langbot rollout restart deployment langbot-gemini-pro
kubectl --context sigsrv-prod -n langbot rollout restart deployment langbot-openai-gpt
