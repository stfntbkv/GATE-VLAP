# Fine-tuning CLIP-RT on Real-World Robot Data

This directory describes how we can fine-tune CLIP-RT on in-domain robot data. 


## Preprocessing Data

We release the in-domain robot data collected by our data collection framework. 

1. Download the in-domain data [here](https://www.dropbox.com/scl/fi/01vcinaxyivbydkziux5e/clip_rt_in_domain_data.tar.gz?rlkey=q8gd63s8zfhdtgj8231u07ff0&st=p2wlyw6i&dl=0) first. 
2. Extact the data `tar -xzvf clip_rt_in_domain_data.tar.gz`
3. Run `python preprocess.py --data-path ./clip_rt_in_domain_data`. This code transforms the json format data to the csv format which is compatible with CLIP-RT training.


## Fine-tuning CLIP-RT via Contrastaive Imitation Learning

After preprocessing in-domain data, simply run `./scripts/finetune.sh`.

Note that you should type the correct path for several arguments in the script (e.g., `--train-data` and `--pretrained`).  





