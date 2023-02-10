from typing import Tuple, List
import hashlib

class Experiment:
    """Experiment class. Contains the logic for assigning users to groups."""

    def __init__(
        self,
        experiment_id: int,
        groups: Tuple[str] = ("A", "B"),
        group_weights: List[float] = None,
    ):
        self.experiment_id = experiment_id
        self.groups = groups
        self.group_weights = group_weights

        # Define the salt for experiment_id.
        # The salt should be deterministic and unique for each experiment_id.
        self.salt = str(experiment_id).encode("utf-8")

        # Define the group weights if they are not provided equaly distributed
        if group_weights is None:
            self.group_weights = [1 / len(groups)] * len(groups)
        else:
            # Check input group weights. They must be non-negative and sum to 1.
            if sum(group_weights) != 1:
                raise ValueError("Group weights must sum to 1.")
            for weight in group_weights:
                if weight < 0:
                    raise ValueError("Group weights must be non-negative.")

    def group(self, click_id: int) -> Tuple[int, str]:
        """Assigns a click to a group.

        Parameters
        ----------
        click_id: int :
            id of the click

        Returns
        -------
        Tuple[int, str] :
            group id and group name
        """

        # Assign the click to a group randomly based on the group weights
        click_id = str(click_id).encode("utf-8")
        hash_input = self.salt + click_id
        hashed = int(hashlib.sha256(hash_input).hexdigest(), 16)
        random_value = hashed / 2**256

        cumulative_probability = 0
        for group_id, weight in enumerate(self.group_weights):
            cumulative_probability += weight
            if random_value < cumulative_probability:
                break

        # Return the group id and group name
        return group_id, self.groups[group_id]
