import torch
import numpy as np
import torch
import cv2


def tensor_normalize(tensor, mean, std):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize.
        mean (tensor or list): mean value to subtract.
        std (tensor or list): std to divide.
    """
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
        tensor = tensor / 255.0
    if type(mean) == list:
        mean = torch.tensor(mean)
    if type(std) == list:
        std = torch.tensor(std)
    tensor = tensor - mean
    tensor = tensor / std
    return tensor


# before 1/(1+exp(-6*(x+1))), after 1/(1+exp(-12*(-x+0.5)))
def compute_time_vector(labels, fps, TT=2, TA=1):
    """
    Compute time vector reflecting time in seconds before or after anomaly range.

    Parameters:
        labels (list or np.ndarray): Binary vector of frame labels (1 for anomalous, 0 otherwise).
        fps (int): Frames per second of the video.
        TT (float): Time-to-anomalous range in seconds (priority).
        TA (float): Time-after-anomalous range in seconds.

    Returns:
        np.ndarray: Time vector for each frame.
    """
    num_frames = len(labels)
    labels = np.array(labels)
    default_value = max(TT, TA) * 2
    time_vector = torch.zeros(num_frames, dtype=float)

    # Get anomaly start and end indices
    anomaly_indices = np.where(labels == 1)[0]
    if len(anomaly_indices) == 0:
        return time_vector  # No anomalies, return all zeros

    # Define maximum frame thresholds for TT and TA
    TT_frames = int(TT * fps)
    TA_frames = int(TA * fps)

    # Iterate through each frame
    for i in range(num_frames):
        if labels[i] == 1:
            time_vector[i] = 0  # Anomalous frame, set to 0
        else:
            # Find distances to the start and end of anomaly ranges
            distances_to_anomalies = anomaly_indices - i

            # Time-to-closest-anomaly-range (TT priority)
            closest_to_anomaly = distances_to_anomalies[distances_to_anomalies > 0]  # After the frame
            if len(closest_to_anomaly) > 0 and closest_to_anomaly[0] <= TT_frames:
                time_vector[i] = -closest_to_anomaly[0] / fps
                continue

            # Time-after-anomaly-range (TA range)
            closest_after_anomaly = distances_to_anomalies[distances_to_anomalies < 0]  # Before the frame
            if len(closest_after_anomaly) > 0 and abs(closest_after_anomaly[-1]) <= TA_frames:
                time_vector[i] = -closest_after_anomaly[-1] / fps
                continue

            # Outside both TT and TA
            time_vector[i] = -100.

    return time_vector


def smooth_labels(labels, time_vector, before_limit=2, after_limit=1):
    xb = before_limit / 2
    xa = after_limit / 2
    kb = 12 / before_limit # 6 for before_limit=2
    ka = 12 / after_limit # 12 for after_limit=1
    sigmoid_before = lambda x: (1 / (1 + torch.exp(-kb * (x + xb)))).float()
    sigmoid_after = lambda x: (1 / (1 + torch.exp(-ka * (-x + xa)))).float()

    before_mask = (time_vector >= -before_limit) & (time_vector < 0)
    after_mask = (time_vector > 0) & (time_vector <= after_limit)

    target_anomaly = (labels == 1).float()
    target_anomaly[before_mask] = sigmoid_before(time_vector[before_mask])
    target_anomaly[after_mask] = sigmoid_after(time_vector[after_mask])
    target_safe = 1 - target_anomaly
    smoothed_target = torch.stack((target_safe, target_anomaly), dim=-1)
    return smoothed_target


def preprocess_fixation_map(image, new_size, gaussian_kernel=120, dilation_kernel=6):
    """Preprocess driver fixation map before resizing."""
    # Step 1: Dilate small fixations at original size
    kernel = np.ones((dilation_kernel, dilation_kernel), np.uint8)
    dilated = cv2.dilate(image, kernel, iterations=10)

    gaussian_kernel = max(1, gaussian_kernel // 2 * 2 + 1)

    # Step 2: Apply Gaussian Blur before resizing
    blurred = cv2.GaussianBlur(dilated, (gaussian_kernel, gaussian_kernel), 0)

    # Step 3: Resize to target dimensions (224x224)
    resized = cv2.resize(blurred, new_size, interpolation=cv2.INTER_LINEAR)

    # Normalize to range [0, 255]
    resized = cv2.normalize(resized, None, 0, 255, cv2.NORM_MINMAX)

    return resized.astype(np.uint8)


class ShortSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_samples_per_epoch, shuffle=True, **kwargs):
        """
        A custom Sampler to limit the number of samples per epoch.

        Args:
            dataset (Dataset): The dataset to sample from.
            num_samples_per_epoch (int): Number of samples per epoch.
            shuffle (bool): Whether to shuffle indices each epoch.
            seed (int): Random seed for reproducibility.
        """
        super().__init__(dataset, **kwargs)
        self.dataset = dataset
        self.num_samples_per_epoch = num_samples_per_epoch
        self.total_size = min(len(dataset), num_samples_per_epoch)
        self.shuffle = shuffle
        self.epoch = None
        #self.set_epoch(0)

    def _generate_indices(self):
        """Generate new random indices for the next epoch using PyTorch's global seed state."""
        if self.shuffle:
            # Use PyTorch's global random generator (no need to reset the seed)
            indices = torch.randperm(len(self.dataset)).numpy()  # Randomly shuffled indices
        else:
            indices = np.arange(len(self.dataset))  # Sequential indices if shuffle=False

        self.indices = indices[:self.num_samples_per_epoch]  # Select subset

    def set_epoch(self, epoch):
        """Allows manual setting of epoch (useful for multi-GPU training)."""
        self.epoch = epoch
        self._generate_indices()  # Refresh indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        """Number of samples per epoch."""
        return self.total_size
