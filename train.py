import argparse
import torch
from pathlib import Path
import torch.optim as optim
from utils.utils import *
from torch.utils.data import DataLoader
from utils.models import *
from tqdm import tqdm
from torchvision.utils import save_image


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--content_dir',     type=str,   default='content_data')
    parser.add_argument('--style_dir',       type=str,   default='style_data')
    parser.add_argument('--vgg',             type=str,   default='vgg_normalised.pth')
    parser.add_argument('--experiment',      type=str,   default='experiment1')
    parser.add_argument('--batch_size',      type=int,   default=2)
    parser.add_argument('--final_size',      type=int,   default=256)
    parser.add_argument('--content_size',    type=int,   default=256)
    parser.add_argument('--style_size',      type=int,   default=256)
    parser.add_argument('--crop',            action='store_true', default=True)
    parser.add_argument('--lr',              type=float, default=1e-4)
    parser.add_argument('--lr_decay',        type=float, default=5e-5)
    parser.add_argument('--epochs',          type=int,   default=10)
    parser.add_argument('--content_weight',  type=float, default=1.0)
    parser.add_argument('--style_weight',    type=float, default=10.0)
    parser.add_argument('--log_interval',    type=int,   default=10)
    parser.add_argument('--save_interval',   type=int,   default=1)
    parser.add_argument('--num_workers',     type=int,   default=0)
    # NEW: how many times to repeat the small dataset per epoch
    parser.add_argument('--repeat',          type=int,   default=100,
                        help='Repeat small dataset N times per epoch (default 100)')
    # NEW: enable augmentation
    parser.add_argument('--augment',         action='store_true', default=True,
                        help='Apply data augmentation (recommended for small datasets)')
    parser.add_argument('--resume',          action='store_true', default=False)
    parser.add_argument('--decoder_path',    type=str,   default=None)
    parser.add_argument('--optimizer_path',  type=str,   default=None)
    return parser.parse_args()


def main():
    args = parse_arguments()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    save_dir = Path('experiments') / args.experiment
    save_dir.mkdir(exist_ok=True, parents=True)

    with open(save_dir / 'args.txt', 'w') as f:
        for key, value in vars(args).items():
            f.write(f'{key}: {value}\n')

    # Transforms — augment=True expands small datasets
    content_transform = get_transform(args.content_size, args.crop, args.final_size, augment=args.augment)
    style_transform   = get_transform(args.style_size,   args.crop, args.final_size, augment=args.augment)

    content_dataset = ImageFolderDataset(args.content_dir, content_transform)
    style_dataset   = ImageFolderDataset(args.style_dir,   style_transform)

    print(f'Content images : {len(content_dataset)}')
    print(f'Style images   : {len(style_dataset)}')

    # Repeat dataset so training sees enough variety each epoch
    content_dataset = RepeatedDataset(content_dataset, repeat=args.repeat)
    style_dataset   = RepeatedDataset(style_dataset,   repeat=args.repeat)

    print(f'After repeat   : {len(content_dataset)} content samples / epoch')
    print(f'After repeat   : {len(style_dataset)} style samples / epoch')

    content_dataloader = DataLoader(
        content_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True
    )
    style_dataloader = DataLoader(
        style_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True
    )

    print(f'Batches per epoch: {min(len(content_dataloader), len(style_dataloader))}')

    encoder = VGGEncoder(args.vgg).to(device)
    decoder = Decoder().to(device)
    optimizer = optim.Adam(decoder.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: 1.0 / (1.0 + args.lr_decay * epoch)
    )

    if args.resume:
        if args.decoder_path is None or args.optimizer_path is None:
            raise ValueError('--decoder_path and --optimizer_path required with --resume')
        decoder.load_state_dict(torch.load(args.decoder_path, map_location=device))
        optimizer.load_state_dict(torch.load(args.optimizer_path, map_location=device))
        print(f'Resumed from {args.decoder_path}')

    mse_loss = torch.nn.MSELoss()
    encoder.eval()

    print('\nTraining...\n')

    for epoch in range(args.epochs):
        running_loss  = 0.0
        running_closs = 0.0
        running_sloss = 0.0
        n_batches     = 0

        progress_bar = tqdm(
            zip(content_dataloader, style_dataloader),
            total=min(len(content_dataloader), len(style_dataloader)),
            desc=f'Epoch [{epoch + 1}/{args.epochs}]'
        )

        for content_batch, style_batch in progress_bar:
            content_batch = content_batch.to(device)
            style_batch   = style_batch.to(device)

            c_feats = encoder(content_batch)
            s_feats = encoder(style_batch)

            t = adapative_instance_normalization(c_feats[-1], s_feats[-1])
            g = decoder(t)

            g_feats = encoder(g)

            loss_c = mse_loss(g_feats[-1], t) * args.content_weight

            loss_s = 0
            for g_f, s_f in zip(g_feats, s_feats):
                g_mean, g_std = calc_mean_std(g_f)
                s_mean, s_std = calc_mean_std(s_f)
                loss_s += mse_loss(g_mean, s_mean) + mse_loss(g_std, s_std)
            loss_s = loss_s * args.style_weight

            loss = loss_c + loss_s

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss  += loss.item()
            running_closs += loss_c.item()
            running_sloss += loss_s.item()
            n_batches     += 1

            progress_bar.set_description(
                f'Epoch [{epoch+1}/{args.epochs}] '
                f'Loss: {loss.item():.4f}  '
                f'C: {loss_c.item():.4f}  '
                f'S: {loss_s.item():.4f}'
            )

            if n_batches % args.log_interval == 0:
                tqdm.write(
                    f'  [Epoch {epoch+1}, Iter {n_batches}]  '
                    f'Avg Loss: {running_loss/n_batches:.4f}  '
                    f'Content: {running_closs/n_batches:.4f}  '
                    f'Style: {running_sloss/n_batches:.4f}'
                )

        scheduler.step()

        avg = running_loss / max(n_batches, 1)
        tqdm.write(f'\nEpoch {epoch+1} complete — Avg Loss: {avg:.4f}\n')

        if (epoch + 1) % args.save_interval == 0 or (epoch + 1) == args.epochs:
            torch.save(decoder.state_dict(),   save_dir / f'decoder_{epoch+1}.pth')
            torch.save(optimizer.state_dict(), save_dir / f'optimizer_{epoch+1}.pth')
            tqdm.write(f'Checkpoint saved → {save_dir}/decoder_{epoch+1}.pth')

            with torch.no_grad():
                output = torch.cat([content_batch, style_batch, g], dim=0)
                save_image(output, save_dir / f'output_{epoch+1}.png', nrow=args.batch_size)
                tqdm.write(f'Sample saved     → {save_dir}/output_{epoch+1}.png')

    print('\nTraining complete.')
    print(f'Final model: {save_dir}/decoder_{args.epochs}.pth')


if __name__ == '__main__':
    main()
