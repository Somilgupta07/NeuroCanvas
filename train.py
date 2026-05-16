import argparse
import torch
from pathlib import Path
from utils.utils import *
from torch.utils.data import DataLoader


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--content_dir',
        type=str,
        default=r'C:\Users\Admin\Desktop\ai\major_projects\NeuroCanvas\content_data',
        help='Location of content dataset'
    )
    parser.add_argument(
    '--batch_size',
    type=int,
    default=4,
    help='Batch size for training'
    )
    parser.add_argument(
        '--style_dir',
        type=str,
        default=r'C:\Users\Admin\Desktop\ai\major_projects\NeuroCanvas\style_data',
        help='Location of style dataset'
    )

    parser.add_argument(
        '--vgg',
        type=str,
        default=r'C:\Users\Admin\Desktop\ai\major_projects\NeuroCanvas\vgg_normalized.pth',
        help='Location of pre-trained VGG'
    )

    parser.add_argument(
        '--final_size',
        type=int,
        default=512,
        help='Size of final image'
    )
    parser.add_argument(
        '--content_size',
        type=int,
        default=256,
        help='Size of content image'
    )
    parser.add_argument(
        '--experiment',
        type=str,
        default='experiment1',
        help='Experiment name'
    )
    parser.add_argument(
        '--style_size',
        type=int,
        default=256,
        help='Size of style image'
    )
    parser.add_argument(
        '--crop',
        action='store_true',
        default=True,
        help='Crop Image'
    )



    return parser.parse_args()


def main():
    args = parse_arguments()

    # Device selection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Create experiment directory
    save_dir = Path('experiments') / args.experiment
    save_dir.mkdir(exist_ok=True, parents=True)

    # Save arguments
    with open(save_dir / 'args.txt', 'w') as args_file:
        for key, value in vars(args).items():
            args_file.write(f'{key}: {value}\n')

    print('Arguments saved successfully!')

    content_transform=get_transform(args.content_size,args.crop,args.final_size)
    style_transform=get_transform(args.style_size,args.crop,args.final_size)
    content_dataset=ImageFolderDataset(args.content_dir,content_transform)
    style_dataset=ImageFolderDataset(args.style_dir,style_transform)

    content_dataloader=DataLoader(content_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  pin_memory=torch.cuda.is_available(),
                                  drop_last=True)
    style_dataloader=DataLoader(style_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  pin_memory=torch.cuda.is_available(),
                                  drop_last=True)
    
    


if __name__ == '__main__':
    main()

