import matplotlib.pyplot as plt

def show_images(dataset, num_images=5):
    """
    Display a sample of images from the dataset.
    """
    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
    for i in range(num_images):
        ax = axes[i]
        img, label = dataset[i]
        ax.imshow(img.permute(1, 2, 0))  # rearrange dimensions for plotting
        ax.set_title(label)
        ax.axis('off')
    plt.show()
    

# get data set stats
'''
print(f"Training set size: {len(train_df)}")
print(f"Validation set size: {len(val_df)}")
print(f"Test set size: {len(test_df)}")


print("Training set class distribution:")
print(train_df['label'].value_counts(normalize=True))

print("Validation set class distribution:")
print(val_df['label'].value_counts(normalize=True))

print("Test set class distribution:")
print(test_df['label'].value_counts(normalize=True))
'''

# get sample images from the dataset
'''
import matplotlib.pyplot as plt

def show_images(dataset, num_images=5):
    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
    for i in range(num_images):
        ax = axes[i]
        img, label = dataset[i]
        ax.imshow(img.permute(1, 2, 0))  # rearrange dimensions for plotting
        ax.set_title(label)
        ax.axis('off')
    plt.show()

# Show images from each dataset
show_images(train_dataset)
show_images(val_dataset)
show_images(test_dataset)
'''


