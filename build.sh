#!/usr/bin/env bash
# exit  on error

set -o errexit

pip install -r requirements.txt

daphne djangito.asgi:application --port=$PORT --bind 0.0.0.0 -v2