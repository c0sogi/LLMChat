#!/bin/bash
git init
git config core.sparseCheckout true
git remote add -f origin git@github.com:c0sogi/api-service.git
echo "app" >> .git/info/sparse-checkout