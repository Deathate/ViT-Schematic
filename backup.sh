#!/bin/bash
# find * -size +1M | cat >> .gitignore
# find * -size +99M | while read -r file; do
#     git lfs track "$file"
# done
cp .gitignore .gitignore.bak
find * -size +99M | cat >> .gitignore
git add -A
git commit -m "Auto commit $(date +%H/%M/%m/%d/%Y)"
git push
mv .gitignore .gitignore_full
mv .gitignore.bak .gitignore
