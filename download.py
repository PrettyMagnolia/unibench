import os

# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from datasets import load_dataset

url_list = [
    # 'clip-benchmark/wds_vtab-resisc45',
    # 'clip-benchmark/wds_vtab-svhn',
    # 'clip-benchmark/wds_vtab-pets',
    # 'clip-benchmark/wds_vtab-eurosat',
    # 'clip-benchmark/wds_vtab-dtd',
    # 'clip-benchmark/wds_vtab-dmlab',
    # "clip-benchmark/wds_vtab-clevr_count_all",
    # "clip-benchmark/wds_vtab-cifar100",
    # "clip-benchmark/wds_vtab-caltech101",
    # "clip-benchmark/wds_vtab-smallnorb_label_elevation",
    # "clip-benchmark/wds_vtab-pcam",
    # "clip-benchmark/wds_vtab-smallnorb_label_azimuth",
    # "clip-benchmark/wds_fer2013",
    # "clip-benchmark/wds_voc2007",
    # "clip-benchmark/wds_country211",
    # "clip-benchmark/wds_fgvc_aircraft",
    # "clip-benchmark/wds_cars",
    # "clip-benchmark/wds_imagenet-a",
    # "clip-benchmark/wds_imagenet-r",
    # "clip-benchmark/wds_imagenetv2",
    # "clip-benchmark/wds_flickr30k",
    # "clip-benchmark/wds_objectnet",
    "clip-benchmark/wds_imagenet1k",


]
for dataset_url in url_list:
    dataset_dir = '/mnt/user_data/yifei/unibench/data/' + dataset_url.split('/')[-1]
    print(dataset_dir, "Start!!!")
    dataset = load_dataset(
        dataset_url,
        trust_remote_code=True,
        split="test",
        num_proc=60,
        cache_dir='/mnt/user_data/yifei/unibench/cache'
    )
    dataset.save_to_disk(str(dataset_dir))
