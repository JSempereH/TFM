from typing import List
import torch
import torch.nn as nn


def are_models_equal(model_list: List[nn.Module]):
    # Get the state_dicts of the models in the list
    state_dicts_list = [model.state_dict() for model in model_list]

    # Check if the keys of state_dicts are the same for all models
    if any(
        set(state_dict.keys()) != set(state_dicts_list[0].keys())
        for state_dict in state_dicts_list[1:]
    ):
        return False

    # Check if the values (weights) of the corresponding keys are the same for all models
    for key in state_dicts_list[0].keys():
        if any(
            not torch.equal(state_dict[key], state_dicts_list[0][key])
            for state_dict in state_dicts_list[1:]
        ):
            return False

    # If all checks pass, the models are equal
    return True


def print_block(sentence):
    # Calculate the length of the sentence for formatting
    sentence_length = len(sentence)

    # Print the top border
    print("#" * (sentence_length + 4))

    # Print the sentence with borders
    print("# " + sentence + " #")

    # Print the bottom border
    print("#" * (sentence_length + 4))
