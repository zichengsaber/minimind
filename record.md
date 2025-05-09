# Start a record

## 测试已有模型

```bash
pip install -r requirements.txt
pip install --upgrade safetensors
sudo apt-get install git-lfs
git clone https://huggingface.co/jingyaogong/MiniMind2
cd ./MiniMind2 && git lfs pull
python eval_model.py --load 1 --model_mode 2
```

## train from scratch

### download data

```bash
pip install modelscope
modelscope download --dataset gongjy/minimind_dataset --local_dir dataset
ln -s /home/aiscuser/.cache/modelscope/hub/datasets/gongjy/minimind_dataset dataset
```