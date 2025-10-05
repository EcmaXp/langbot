#!/usr/bin/env bash
kubectl --context sigsrv-prod -n langbot apply -f langbot-vault.yaml
kubectl --context sigsrv-prod -n langbot apply -f langbot-policy.yaml
kubectl --context sigsrv-prod -n langbot apply -f langbot-claude-sonnet.yaml
kubectl --context sigsrv-prod -n langbot apply -f langbot-gemini-pro.yaml
kubectl --context sigsrv-prod -n langbot apply -f langbot-openai-gpt.yaml
./restart-langbot.sh
