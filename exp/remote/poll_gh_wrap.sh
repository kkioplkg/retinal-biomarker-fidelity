#!/usr/bin/env bash
cd /e/Programming/research_ws/medical1/exp
exec bash remote/poll_gh.sh >> logs/poll_gh.log 2>&1
