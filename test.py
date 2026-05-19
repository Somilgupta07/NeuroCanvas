import argparse
from pathlib import Path
import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

from utils.models import VGGEncoder, Decoder
from utils.utils import adapative_instance_normalization


def load_image(path: str, size: int = 512) -> torch.Tensor:
    """Load a single image as a [1, 3, H, W] tensor, values in [0, 1]."""
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
    ])
    img = Image.open(path).convert('RGB')
    return transform(img).unsqueeze(0)   # add batch dim


def stylize(
    content_tensor: torch.Tensor,
    style_tensor: torch.Tensor,
    encoder: VGGEncoder,
    decoder: Decoder,
    alpha: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Run AdaIN style transfer.

    alpha = 1.0  → full stylization (default)
    alpha = 0.0  → original content unchanged
    """
    content_tensor = content_tensor.to(device)
    style_tensor   = style_tensor.to(device)

    with torch.no_grad():
        c_feats = encoder(content_tensor)          # returns (h1, h2, h3, h4)
        s_feats = encoder(style_tensor)

        # AdaIN applied at relu4-1 (last feature map)
        t = adapative_instance_normalization(c_feats[-1], s_feats[-1])

        # Alpha blending: interpolate between content features and AdaIN output
        t = alpha * t + (1.0 - alpha) * c_feats[-1]

        # Decode back to image space — Decoder already outputs [0, 1] via Sigmoid
        output = decoder(t)

    return output.cpu()


def parse_arguments():
    parser = argparse.ArgumentParser(description='AdaIN Neural Style Transfer — Inference')

    # Single image mode
    parser.add_argument('--content',  type=str, default=None, help='Path to a single content image')
    parser.add_argument('--style',    type=str, default=None, help='Path to a single style image')
    parser.add_argument('--output',   type=str, default='output.png', help='Output file path')

    # Batch mode
    parser.add_argument('--content_dir', type=str, default=None, help='Folder of content images')
    parser.add_argument('--style_dir',   type=str, default=None, help='Folder of style images')
    parser.add_argument('--output_dir',  type=str, default='results', help='Output folder for batch mode')

    # Model paths
    parser.add_argument('--vgg',     type=str, required=True, help='Path to vgg_normalised.pth')
    parser.add_argument('--decoder', type=str, required=True, help='Path to trained decoder checkpoint')

    # Settings
    parser.add_argument('--alpha',        type=float, default=1.0,
                        help='Style strength: 0.0 = content only, 1.0 = full style (default: 1.0)')
    parser.add_argument('--content_size', type=int,   default=512, help='Resize content image to this size')
    parser.add_argument('--style_size',   type=int,   default=512, help='Resize style image to this size')

    return parser.parse_args()


def main():
    args = parse_arguments()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load models
    encoder = VGGEncoder(args.vgg).to(device)
    decoder = Decoder().to(device)
    decoder.load_state_dict(torch.load(args.decoder, map_location=device))
    decoder.eval()
    encoder.eval()
    print(f'Decoder loaded from: {args.decoder}')

 
    if args.content and args.style:
        content = load_image(args.content, args.content_size)
        style   = load_image(args.style,   args.style_size)

        output = stylize(content, style, encoder, decoder, args.alpha, device)

        save_image(output, args.output)
        print(f'Saved stylized image → {args.output}')


    elif args.content_dir and args.style_dir:
        IMG_EXTS = ('.jpg', '.jpeg', '.png')
        content_paths = sorted(p for p in Path(args.content_dir).iterdir() if p.suffix.lower() in IMG_EXTS)
        style_paths   = sorted(p for p in Path(args.style_dir).iterdir()   if p.suffix.lower() in IMG_EXTS)

        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f'Content images: {len(content_paths)}')
        print(f'Style images:   {len(style_paths)}')

        for c_path in content_paths:
            content = load_image(str(c_path), args.content_size)
            for s_path in style_paths:
                style  = load_image(str(s_path), args.style_size)
                output = stylize(content, style, encoder, decoder, args.alpha, device)

                out_name = f'{c_path.stem}__{s_path.stem}.png'
                save_image(output, out_dir / out_name)
                print(f'  Saved → {out_dir / out_name}')

        print(f'\nAll results saved to: {out_dir}')

    else:
        print('ERROR: Provide either --content + --style, or --content_dir + --style_dir.')
        print('Run  python test.py --help  for usage.')


if __name__ == '__main__':
    main()
