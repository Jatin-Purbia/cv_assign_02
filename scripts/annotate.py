"""Manual polygon annotation with Matplotlib only. Click boundary, Enter saves; close cancels."""
from src import runtime
runtime.configure()
import argparse
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector
import torch
from src.preprocessing import read_image


def polygon_mask(vertices: list[tuple[float, float]]) -> torch.Tensor:
    """Even-odd ray crossing rasterization at pixel centers (no morphology/drawing API)."""
    yy, xx = torch.meshgrid(torch.arange(256)+.5, torch.arange(256)+.5, indexing='ij')
    inside = torch.zeros((256, 256), dtype=torch.bool)
    for i, (x1, y1) in enumerate(vertices):
        x2, y2 = vertices[(i+1) % len(vertices)]
        if y1 != y2:
            crossing = ((y1 > yy) != (y2 > yy)) & (xx < (x2-x1)*(yy-y1)/(y2-y1)+x1)
            inside ^= crossing
    return inside


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image', required=True)
    p.add_argument('--root', default='captured')
    args = p.parse_args()
    source, root = Path(args.image), Path(args.root)
    for folder in ('original', 'images_256', 'masks_256'):
        (root/folder).mkdir(parents=True, exist_ok=True)
    import shutil
    dest = root/'original'/source.name
    if source.resolve() != dest.resolve():
        if dest.exists():
            raise FileExistsError(dest)
        shutil.copyfile(source, dest)
    image = read_image(source)
    cv2.imwrite(str(root/'images_256'/f'{source.stem}.png'), image)
    fig, ax = plt.subplots()
    ax.imshow(image[:, :, ::-1])
    ax.set_title('Click polygon vertices; finish polygon, then Enter to save. Close to cancel.')
    vertices = []
    def selected(points):
        vertices[:] = points
    selector = PolygonSelector(ax, selected, useblit=True)
    def key(event):
        if event.key == 'enter' and len(vertices) >= 3:
            mask = polygon_mask(vertices).to(torch.uint8).numpy()*255
            target = root/'masks_256'/f'{source.stem}.png'
            if not cv2.imwrite(str(target), mask):
                raise OSError(target)
            plt.close(fig)
    fig.canvas.mpl_connect('key_press_event', key)
    plt.show()


if __name__ == '__main__':
    main()
