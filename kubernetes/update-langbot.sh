#!/usr/bin/env bash
kubectl --context sigsrv-prod -n langbot apply -f langbot-vault.yaml
kubectl --context sigsrv-prod -n langbot apply -f langbot-policy.yaml
kubectl --context sigsrv-prod -n langbot apply -f langbot-claude.yaml
kubectl --context sigsrv-prod -n langbot apply -f langbot-openai-gpt-4o.yaml
kubectl --context sigsrv-prod -n langbot apply -f langbot-openai-o1.yaml
./restart-langbot.sh
