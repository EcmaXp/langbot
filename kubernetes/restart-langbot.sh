#!/usr/bin/env bash
docker buildx build --platform linux/amd64 -t registry-t1.deer-neon.ts.net/langbot:latest .. --push
kubectl --context sigsrv-t1 -n langbot rollout restart deployment langbot-claude-opus
kubectl --context sigsrv-t1 -n langbot rollout restart deployment langbot-gemini-pro
kubectl --context sigsrv-t1 -n langbot rollout restart deployment langbot-openai-gpt
