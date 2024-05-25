#!/usr/bin/env fish
set OP_LANGBOT_SECRET_PATH "op://sigsrv-prod/sigsrv-prod-langbot-secrets"

kubectl create ns --context sigsrv-prod langbot

set OP_LANGBOT_40_SECRET_PATH "$OP_LANGBOT_SECRET_PATH/langbot-secret-gpt-4o"
kubectl create secret generic --context sigsrv-prod -n langbot langbot-secret-gpt-4o \
  --from-literal=DISCORD_TOKEN=(op read "$OP_LANGBOT_SECRET_PATH/gpt-4o/DISCORD_TOKEN") \
  --from-literal=OPENAI_API_KEY=(op read "$OP_LANGBOT_SECRET_PATH/gpt-4o/OPENAI_API_KEY")
kubectl apply --context sigsrv-prod -n langbot -f langbot-gpt-4o.yaml

kubectl create secret generic --context sigsrv-prod -n langbot langbot-secret-claude \
  --from-literal=DISCORD_TOKEN=(op read "$OP_LANGBOT_SECRET_PATH/claude/DISCORD_TOKEN") \
  --from-literal=ANTHROPIC_API_KEY=(op read "$OP_LANGBOT_SECRET_PATH/claude/ANTHROPIC_API_KEY") \
  --from-literal=OPENAI_API_KEY=(op read "$OP_LANGBOT_SECRET_PATH/claude/OPENAI_API_KEY")
kubectl apply --context sigsrv-prod -n langbot -f langbot-claude.yaml

kubectl create secret generic --context sigsrv-prod -n langbot langbot-secret-gemini \
  --from-literal=DISCORD_TOKEN=(op read "$OP_LANGBOT_SECRET_PATH/gemini/DISCORD_TOKEN") \
  --from-literal=GOOGLE_API_KEY=(op read "$OP_LANGBOT_SECRET_PATH/gemini/GOOGLE_API_KEY") \
  --from-literal=OPENAI_API_KEY=(op read "$OP_LANGBOT_SECRET_PATH/gemini/OPENAI_API_KEY")
kubectl apply --context sigsrv-prod -n langbot -f langbot-gemini.yaml
