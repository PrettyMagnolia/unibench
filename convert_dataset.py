import os
from datasets import Dataset
from tqdm import tqdm
import argparse
from PIL import Image
from unibench.common_utils import DATA_DIR, args


for dataset_name in args.dataset_names:
    source_dir = DATA_DIR.joinpath(dataset_name)
    target_dir = DATA_DIR.parent.joinpath('images', dataset_name)
    print(f'Processing dataset {dataset_name}')
    os.makedirs(target_dir, exist_ok=True)
    dataset = Dataset.load_from_disk(source_dir)
    for data in tqdm(dataset):
        # image_data = data.get('webp') or data.get('0.webp') or data.get('jpg') or data.get('png')

        samples = []
        for key in data.keys():
            if 'webp' in key or 'jpg' in key or 'png' in key:
                image_data = data[key]
                if image_data.mode != 'RGB':
                    image_data = image_data.convert('RGB')
                samples.append(image_data)
                
        if len(samples) == 0:
            raise ValueError("No valid image data found in the dictionary.")
        
        if len(samples) > 1:
            for idx, image_data in enumerate(samples):
                name = data['__key__'] + '_' + str(idx)
                image_data.save(target_dir.joinpath(f'{name}.jpg'), format='JPEG')
        else:
            image_data = samples[0]
            name = data['__key__']
            image_data.save(target_dir.joinpath(f'{name}.jpg'), format='JPEG')
    print(f'Finished processing dataset {dataset_name} from {source_dir} to {target_dir}')
        
