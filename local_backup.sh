# rsync -vzhr --delete --progress ../ViT-Schematic server7:/home/deathate/Projects
rsync -vzhr --delete --progress ../ViT-Schematic server5:/home/112/deathate/Projects --exclude-from='exclude-list.txt'
# rsync -vzhr  --progress server7:/home/deathate/Projects/ViT-Schematic/tmp . --dry-run