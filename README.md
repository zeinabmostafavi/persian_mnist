# Persian_mnist

## install

pip install requirements.txt

## Train
To train the model, please run fallowing:
```
python3 train.py --device cuda --data_path  dataset
```
## Test
To test the model, please run fallowing:
```
python test.py --device cuda  --weight weight/mnist_persian.pth
```
## Inference
To inference the model, please run fallowing:
```

python inference.py --device cuda --weight weight/mnist_persian.pth  --image_path input/5.jpg


```

