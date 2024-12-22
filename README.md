# splatgrasp

## Installation
```
cd splatbot
docker build -t drake .
```

## Run Docker
Run the following: 
```
docker run -it --network host -v $(pwd):/workspace -w /workspace drake 
```
Alternatively, run
```
sudo apachectl start
docker run -it -p 8000:7000 -v $(pwd):/workspace -w /workspace drake
```
This sets up the simulation on port `8000`.

