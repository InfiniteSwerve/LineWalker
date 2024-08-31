from PIL import Image
import torch
from torch import Tensor
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pandas as pd
import einops
import numpy as np

"""
- get images
- get point differences
- set up data pipeline
    - pass pairs of images
    - ensure image size is at least same size as point jumps
- Maybe only extract sections of images without opening the whole thing?
"""


# TODO: generalize this to handle things without jpgs
def get_padded_image(file_path: str, padding) -> Tensor:
    image = Image.open(file_path)
    to_tensor = transforms.ToTensor()
    image_tensor = to_tensor(image)
    padded_image_tensor = torch.nn.functional.pad(
        image_tensor, (padding, padding, padding, padding), "constant", value=1
    )
    return padded_image_tensor


def extract_patch(image: Tensor, position, size, padding, reshape=True) -> Tensor:
    position = position.round().int()
    x = position[0]
    y = position[1]
    x += padding
    y += padding
    size = int(size)
    t = image[:, y - size : y + size, x - size : x + size]
    if reshape:
        t = einops.rearrange(t, "colors x_dim y_dim -> (colors x_dim y_dim)")
    return t


# TODO: size and patches
def extract_patches(image: Tensor, positions: Tensor, padding, size, reshape=True):
    positions = positions.round().int() + padding
    x1 = int(positions[0, 0])
    y1 = int(positions[0, 1])
    x2 = int(positions[1, 0])
    y2 = int(positions[1, 1])
    t1 = image[:, y1 - size : y1 + size, x1 - size : x1 + size]
    t2 = image[:, y2 - size : y2 + size, x2 - size : x2 + size]
    patches = torch.stack([t1, t2])

    if reshape:
        patches = einops.rearrange(
            patches, "seq colors x_dim y_dim -> seq (colors x_dim y_dim)"
        )
    return patches


def generate_training_data(image, data, size, vectorize=True):
    data = data.round()
    return torch.stack(
        [extract_patches(image, positions, size, size, vectorize) for positions in data]
    )


# TODO: change this from hardcoded csv
def get_line(index: int) -> Tensor:
    # returns a [point_count, 2] tensor
    df = pd.read_csv("juneau_points.csv", names=["index", "x", "y"])
    points = df[df["index"] == index]
    x_tensor = torch.tensor(points["x"].values)
    y_tensor = torch.tensor(points["y"].values)
    return torch.stack([x_tensor, y_tensor], dim=1)


# TODO: change this from hardcoded csv
def get_line_data(
    indices: list[int], generate_targets=True, flattened=True, split=True
) -> Tensor:
    output = []
    for i in indices:
        # TODO: will read the csv lots of times, don't do this.
        line = get_line(i)
        jumps = line[1:] - line[:-1]
        if generate_targets:
            temp = list(
                zip(
                    torch.stack([line[:-1], line[1:]], dim=1),
                    torch.cat([jumps[1:], torch.zeros(1, 2)]),
                )
            )
        else:
            temp = torch.stack([line[:-2], line[1:-1]], dim=1)
        output.append(temp)
    if flattened:
        output = [c for l in output for c in l]
        if split:
            output = torch.utils.data.random_split(output, [0.8, 0.2])
    return output


def get_answer_triplet(image, datum, size, padding):
    input_ims = extract_patches(image, datum[0], size, reshape=False)
    answer_im = extract_patch(
        image, datum[1] + datum[0][1], size, padding, reshape=False
    )
    return input_ims, answer_im


def to_im(patch):
    return einops.rearrange(patch, "c h w -> h w c")


def vec_to_im(patch):
    return einops.rearrange(
        patch, "(c h w) -> c h w", c=3, h=int((list(patch.shape)[0] // 3) ** 0.5)
    )


def numpy(v):
    return v.detach().cpu().numpy()


def compare_model_answer(image, model, datum, size, padding):
    input_ims = extract_patches(image, datum[0], padding, size, reshape=False)
    print(image.shape)
    print(datum[1].shape)
    print(datum[0][1].shape)
    answer_im = extract_patch(
        image, datum[1] + datum[0][1], size, padding, reshape=False
    )
    model_input = einops.rearrange(input_ims, "seq color h w -> 1 seq (color h w)")
    model_ans = model(model_input)
    model_ans_position = datum[0][1] + model_ans[0, -1].round().int()
    model_image = extract_patch(image, model_ans_position, size, padding, reshape=False)

    patch1 = numpy(input_ims[0])
    patch2 = numpy(input_ims[1])
    ans = numpy(answer_im)
    guess = numpy(model_image)
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))  # 1 row, 4 columns
    im_names = ["Input 1", "Input 2", "Answer", "Model Guess"]
    patches = [patch1, patch2, ans, guess]

    for i, ax in enumerate(axs):
        ax.imshow(to_im(patches[i]))
        ax.set_title(im_names[i])

        # Calculate the center coordinates
        H = W = patches[i].shape[
            1
        ]  # Assuming patches are 2D or have shape (H, W, Channels)

        # Draw an 'x' at the center
        ax.plot(W / 2, H / 2, marker="x", markersize=10, color="red")

        ax.axis("off")

    plt.show()


def compare_model_answer_with_actual(model, datum, size):
    with torch.inference_mode():
        model_ans = (
            model(datum[0].to("cuda").unsqueeze(0))[0, -1].cpu().numpy().squeeze()
        )
    r = lambda p: einops.rearrange(
        p, "(c h w) -> c h w", c=3, h=int((list(p.shape)[0] // 3) ** 0.5)
    )
    patch1 = numpy(r(datum[0][0]))
    patch2 = numpy(r(datum[0][1]))
    ans = numpy(datum[1]).squeeze()

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))  # 1 row, 2 columns
    ax[1].imshow(to_im(patch1))
    ax[0].imshow(to_im(patch2))
    arrowprops = dict(arrowstyle="->", alpha=0.5, linewidth=2, color="red")
    base = [size, size]

    ax[1].annotate("", xy=base + model_ans, xytext=base, arrowprops=arrowprops)
    arrowprops = dict(arrowstyle="->", alpha=0.5, linewidth=2, color="blue")
    ax[1].annotate("", xy=base + ans, xytext=base, arrowprops=arrowprops)
    print(f"Ans: {ans}")
    print(f"Model ans: {model_ans}")

    plt.show()


# Counter Clockwise
def rotate_and_flip_image(image, angle=0, flip=False):
    if flip:
        image = image.flip(-2)

    if angle == 0:
        return image
    elif angle == 90:
        return image.transpose(-1, -2).flip(-1)
    elif angle == 180:
        return image.flip(-1).flip(-2)
    elif angle == 270:
        return image.flip(-1).transpose(-1, -2)
    else:
        raise ValueError("Unsupported angle. Supported values are 0, 90, 180, 270.")


# Counter Clockwise
def rotate_and_flip_coordinates(coords, angle=0, flip=False):
    angle = np.radians(angle)
    x = coords[0]
    y = coords[1]
    if flip:
        y = y * -1
    x1 = x * np.cos(angle) - y * np.sin(angle)
    y1 = x * np.sin(angle) + y * np.cos(angle)
    return torch.stack([x1, y1])


"""
I should be able to pass a single function some arguments and get back either normal or data augmnented training and test sets
Input:
- File name
- line indices
- rotations = True
- Flip = True
- train/test split

Output:
list: 2
    |_ list: lines * points * rotations * flips
        |_ tuple
            |_ Tensor (dtype: torch.float64)
            |   |__ dim_0 (c*h*w)
            |_ Tensor(dtype: torch.float64) 
                |_ dim_0 (2)
"""


# TODO: Abstract over line csv
# TODO: abstract over data augmentation functions and inputs
def generate_all_data(
    file: str,
    indices: list[int],
    padding: int,
    window_size: int,
    rotations=True,
    flip=True,
    split=True,
    train_test_split=0.8,
):
    image: Tensor = get_padded_image(file, padding)
    line_data: Tensor = get_line_data(indices, split=False)
    # apply train/test split later on
    angles = [0]
    flips = [False]
    if rotations:
        angles = [0, 90, 180, 270]
    if flip:
        flips = [True, False]

    augmented_data = []
    for data in line_data:
        coordinates = data[0]
        jump = data[1]
        original_images = extract_patches(
            image,
            coordinates,
            padding,
            window_size,
            reshape=False,
        )
        for angle in angles:
            for fl in flips:
                local_ims: list[Tensor] = []
                for local_im in original_images:
                    if angle != 0 or flip != False:
                        local_ims.append(rotate_and_flip_image(local_im, angle, fl))
                    else:
                        local_ims.append(local_im)

                coords = rotate_and_flip_coordinates(jump, angle, fl)
                local_ims = einops.rearrange(
                    torch.stack(local_ims), "seq c h w -> seq (c h w)"
                )
                augmented_data.append((local_ims, coords))
    if split:
        train_sub, test_sub = torch.utils.data.random_split(
            augmented_data, [train_test_split, 1 - train_test_split]
        )
        xs_train, ys_train = map(list, zip(*train_sub))
        xs_test, ys_test = map(list, zip(*test_sub))
        return xs_train, ys_train, xs_test, ys_test

    return augmented_data


"""
get line
get point jumps <- MSE target

Transformer Takes:
- Image at previous 2 points as vectors, so 2 vectors total
Transformer Returns:
- Returns jump coordinates from current position

MSE takes:
- Transformer output
- Next point jump

If we're at the end of the line, the correct next jump is (0,0), which will be the signal to stop the walking the line

"""
