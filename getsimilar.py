import pickle
import os
import sys
from numpy.testing import assert_almost_equal

def get_names(name, similar_names, similar_values):
    print("Getting similar images")
    images = list(similar_names.loc[name, :])
    values = list(similar_values.loc[name, :])
    if name in images:
        assert_almost_equal(max(values), 1, decimal = 5)
        images.remove(name)
        values.remove(max(values))
    
    # Filter to only existing images
    existing_images = []
    existing_values = []
    for img, val in zip(images, values):
        # Accept image if present in either images/ or resized/
        if os.path.exists(os.path.join("images", img)) or os.path.exists(os.path.join("resized", img)):
            existing_images.append(img)
            existing_values.append(val)
    
    return name, existing_images[:8], existing_values[:8]


def get_images(input_image, model):
    similar_names = pickle.load(open(os.path.join("pickles", model, "similar_names.pkl"), 'rb'))
    similar_values = pickle.load(open(os.path.join("pickles", model, "similar_values.pkl"), 'rb'))
    
    if input_image in set(similar_names.index):
        return get_names(input_image, similar_names, similar_values)
    elif input_image+".png" in set(similar_names.index):
        input_image = input_image+".png"
        return get_names(input_image, similar_names, similar_values)
    elif input_image+".jpg" in set(similar_names.index):
        input_image = input_image+".jpg"
        return get_names(input_image, similar_names, similar_values)
    else:
        print("'{}' is not in images.\nMake sure the name of the image is in the format: artist-name_image-name[.png or .jpg]\nFor example, for The Starry Night by Van Gogh: van-gogh_the-starry-night".format(input_image))
        sys.exit(2)