import cv2
import numpy as np

# Function to add Gaussian noise to an image


def add_gaussian_noise(image, mean=0, stddev=10):
    noise = np.random.normal(mean, stddev, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return noisy_image

# Function to add salt-and-pepper noise to an image


def add_salt_and_pepper_noise(image, salt_prob=0.05, pepper_prob=0.05):
    noisy_image = np.copy(image)
    height, width = image.shape[:2]
    salt_count = int(height * width * salt_prob)
    pepper_count = int(height * width * pepper_prob)

    # Add salt noise
    salt_coords = [np.random.randint(0, height - 1, salt_count),
                   np.random.randint(0, width - 1, salt_count)]
    salt_coords = np.clip(salt_coords, 0, height - 1)
    noisy_image[salt_coords[0], salt_coords[1]] = 255

    # Add pepper noise
    pepper_coords = [np.random.randint(0, height - 1, pepper_count),
                     np.random.randint(0, width - 1, pepper_count)]
    pepper_coords = np.clip(pepper_coords, 0, height - 1)
    noisy_image[pepper_coords[0], pepper_coords[1]] = 0

    return noisy_image

# Function to remove noise using mean filter


def remove_noise_mean(image, kernel_size=3):
    filtered_image = cv2.blur(image, (kernel_size, kernel_size))
    return filtered_image

# Function to remove noise using median filter


def remove_noise_median(image, kernel_size=3):
    filtered_image = cv2.medianBlur(image, kernel_size)
    return filtered_image


# Load an image
image = cv2.imread('./beagle-hound-dog.jpg', cv2.IMREAD_GRAYSCALE)

# Add Gaussian noise to the image
noisy_image_gaussian = add_gaussian_noise(image)

# Add salt-and-pepper noise to the image
noisy_image_salt_pepper = add_salt_and_pepper_noise(image)

# Remove noise using mean filter
filtered_image_mean_gaussian = remove_noise_mean(noisy_image_gaussian)
filtered_image_mean_salt_pepper = remove_noise_mean(noisy_image_salt_pepper)

# Remove noise using median filter
filtered_image_median_gaussian = remove_noise_median(noisy_image_gaussian)
filtered_image_median_salt_pepper = remove_noise_median(
    noisy_image_salt_pepper)

# Display the original image and the filtered images
cv2.imshow('Original Image', image)
cv2.imshow('Noisy Image (Gaussian)', noisy_image_gaussian)
cv2.imshow('Noisy Image (Salt and Pepper)', noisy_image_salt_pepper)
cv2.imshow('Filtered Image (Mean, Gaussian)', filtered_image_mean_gaussian)
cv2.imshow('Filtered Image (Mean, Salt and Pepper)',
           filtered_image_mean_salt_pepper)
cv2.imshow('Filtered Image (Median, Gaussian)', filtered_image_median_gaussian)
cv2.imshow('Filtered Image (Median, Salt and Pepper)',
           filtered_image_median_salt_pepper)

# Wait for key press and then close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()
