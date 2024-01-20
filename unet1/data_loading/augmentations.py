def get_augmentation_funcs(augmentation_params):
    functions = []
    # TODO: define augmentation functions here
    return functions

def apply_augmentations(data,augmentation_funcs):
    augmented_data=data
    for augmentation_func in augmentation_funcs:
        augmented_data=augmentation_func(augmented_data[0],augmented_data[1])
    return augmented_data