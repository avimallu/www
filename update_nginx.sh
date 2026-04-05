hugo && rsync -av --delete public/ vps:~/www
# Also might need
ssh vps "chmod -R o+rX www/"

