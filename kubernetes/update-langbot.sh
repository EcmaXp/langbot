#!/usr/bin/env bash
kubectl --context sigsrv-prod -n langbot apply -f langbot-policy.yaml
kubectl --context sigsrv-prod -n langbot apply -f langbot-claude.yaml
kubectl --context sigsrv-prod -n langbot apply -f langbot-gemini.yaml
kubectl --context sigsrv-prod -n langbot apply -f langbot-gpt-4o.yaml
./restart-langbot.sh
