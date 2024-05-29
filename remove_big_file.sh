find * -size +30M | while read -r file; do
    git rm  --cached  $file
done
