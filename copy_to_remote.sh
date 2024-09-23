zip -r "$1.zip" "$1"
scp -r "$1.zip" aaron:"C:\Users\hoaro\Downloads"
rm -r "$1.zip"