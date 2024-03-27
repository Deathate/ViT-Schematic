#!/bin/bash
# find * -size +1M | cat >> .gitignore
find * -size +99M | while read -r file; do
    git lfs track "$file"
done
git add .
git commit -m "Auto commit $(date +%H/%M/%m/%d/%Y)"
git push

