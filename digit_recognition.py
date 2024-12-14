import cv2
import numpy as np
import os

def load_images_from_directory(
    base_path: str, target_size: tuple[int, int] = (28, 28), output_dir: str = "processed"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load and preprocess images from a directory with subfolders named '0' to '9'.

    :param base_path: Path to the dataset directory.
    :param target_size: Target size (height, width) for resizing images.
    :param output_dir: Path to save processed images.
    :return: Tuple of images and labels as numpy arrays.
    """
    images, labels = [], []

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    for label in range(10):  # Assuming subfolders are named '0' to '9'
        folder_path = os.path.join(base_path, str(label))
        output_folder = os.path.join(output_dir, str(label))
        os.makedirs(output_folder, exist_ok=True)

        if not os.path.exists(folder_path):
            print(f"Warning: Directory {folder_path} does not exist. Skipping.")
            continue

        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            try:
                img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                if img is None:
                    raise ValueError(f"Invalid image: {img_path}")

                processed_img = preprocess_image(img, target_size)
                output_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_path, processed_img)

                images.append(processed_img.flatten())
                labels.append(label)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32)


def preprocess_image(img: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """
    Preprocess an image by removing alpha channel, converting to grayscale, applying adaptive thresholding, 
    and resizing it while maintaining aspect ratio.

    :param img: Input image.
    :param target_size: Target size for the output image.
    :return: Preprocessed image.
    """
    # Remove alpha channel if present
    if img.shape[-1] == 4:  # Image has an alpha channel
        img = remove_alpha_channel(img)

    # Convert to grayscale if not already
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding
    block_size = max(3, (min(img.shape) // 10) | 1)  # Ensure odd block size
    img_bin = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, 2)

    # Resize while maintaining aspect ratio
    return resize_with_aspect_ratio(img_bin, target_size)


def remove_alpha_channel(img: np.ndarray) -> np.ndarray:
    """
    Remove alpha channel and replace transparent pixels with white.

    :param img: Input image with alpha channel.
    :return: RGB image without alpha channel.
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    img_rgb[img[:, :, 3] == 0] = [255, 255, 255]
    return img_rgb


def resize_with_aspect_ratio(image: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """
    Resize an image while maintaining the aspect ratio.

    :param image: Input image (grayscale).
    :param target_size: Tuple (height, width) of the target size.
    :return: Resized image centered on a white canvas.
    """
    h, w = image.shape
    scale = min(target_size[0] / h, target_size[1] / w)
    new_w, new_h = int(w * scale), int(h * scale)

    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full(target_size, 255, dtype=np.uint8)
    y_offset = (target_size[0] - new_h) // 2
    x_offset = (target_size[1] - new_w) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_image
    return canvas


def normalize_data(images: np.ndarray) -> np.ndarray:
    """
    Normalize image data to the range [0, 1].

    :param images: Input image data.
    :return: Normalized image data.
    """
    return images / 255.0


def split_data(images: np.ndarray, labels: np.ndarray, test_size: float = 0.2, random_seed: int = 42) -> tuple:
    """
    Split data into training and testing sets.

    :param images: Array of images.
    :param labels: Array of labels.
    :param test_size: Proportion of data to use for testing.
    :param random_seed: Random seed for reproducibility.
    :return: Train-test split of images and labels.
    """
    np.random.seed(random_seed)
    indices = np.arange(len(labels))
    np.random.shuffle(indices)

    split_idx = int(len(indices) * (1 - test_size))
    train_indices, test_indices = indices[:split_idx], indices[split_idx:]

    return images[train_indices], images[test_indices], labels[train_indices], labels[test_indices]


def evaluate_model(results: np.ndarray, true_labels: np.ndarray) -> float:
    """
    Evaluate model accuracy and display a confusion matrix.

    :param results: Predicted labels.
    :param true_labels: True labels.
    :return: Accuracy as a percentage.
    """
    accuracy = np.mean(results.flatten() == true_labels) * 100
    print(f"Accuracy: {accuracy:.2f}%")

    # Confusion matrix
    unique_labels = np.unique(true_labels)
    confusion_matrix = np.zeros((len(unique_labels), len(unique_labels)), dtype=np.int32)
    for true, pred in zip(true_labels, results.flatten()):
        confusion_matrix[int(true), int(pred)] += 1

    print("Confusion Matrix:")
    print(confusion_matrix)
    return accuracy


def main():
    dataset_path = "C:/Users/mary jane tabang/Desktop/digit/DATA"  # Replace with your dataset's path
    processed_path = "processed"

    print("Loading and preprocessing dataset...")
    images, labels = load_images_from_directory(dataset_path, output_dir=processed_path)
    images = normalize_data(images)
    print(f"Loaded {len(images)} images and saved to {processed_path}/.")

    train_images, test_images, train_labels, test_labels = split_data(images, labels)

    print("Training k-NN classifier...")
    knn = cv2.ml.KNearest_create()
    knn.setDefaultK(5)
    knn.setAlgorithmType(cv2.ml.KNearest_BRUTE_FORCE)

    knn.train(train_images, cv2.ml.ROW_SAMPLE, train_labels)
    print("Training complete.")

    model_filename = "knn_model.xml"
    knn.save(model_filename)
    print(f"Model saved to {model_filename}")

    print("Testing the model...")
    _, results, _, _ = knn.findNearest(test_images, k=5)
    evaluate_model(results, test_labels)


if __name__ == "__main__":
    main()
