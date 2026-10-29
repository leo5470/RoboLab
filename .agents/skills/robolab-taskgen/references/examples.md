# Task Generation Examples

Source patterns, not proof that the named scene/assets are valid. Resolve each actual scene, verify prim names and function signatures, and attach validation evidence. For an unqualified draft, state pending checks and follow the output-location rules in SKILL.md.

## Example 1: Simple Pick-and-Place

**Goal:** Pick up a banana and place it in a bowl.

- **Conditional:** `object_in_container` -- single object into an open-top container
- **Subtask:** `pick_and_place` composite -- tracks grab and placement
- **Attributes:** `semantics` -- requires recognizing "banana" and "bowl"

```python
from dataclasses import dataclass

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.core.scenes.utils import import_scene
from robolab.core.task.conditionals import object_in_container, pick_and_place
from robolab.core.task.task import Task

@configclass
class BananaInBowlTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_in_container,
        params={
            "object": "banana",
            "container": "bowl",
            "gripper_name": "gripper",
            "tolerance": 0.0,
            "require_contact_with": True,
            "require_gripper_detached": True,
        },
    )


@dataclass
class BananaInBowlTask(Task):
    contact_object_list = ["banana", "bowl", "table"]
    scene = import_scene("banana_bowl.usda", contact_object_list)
    terminations = BananaInBowlTerminations
    instruction = {
        "default": "Pick up the banana and place it in the bowl",
        "vague": "Put the fruit in the bowl",
        "specific": "Grasp the yellow banana and place it inside the bowl on the table",
    }
    episode_length_s: int = 50
    attributes = ['semantics']
    subtasks = [
        pick_and_place(
            object=["banana"], container="bowl", logical="all", score=1.0,
            placement_func=object_in_container,
            placement_params={"gripper_name": "gripper", "tolerance": 0.0,
                              "require_contact_with": True, "require_gripper_detached": True},
        )
    ]
```

## Example 2: Multi-Object Sorting

**Goal:** Put coffee_can in bin_a06 and mustard and sugar_box in bin_b03, in any order.

- **Conditional:** `object_groups_in_containers` -- multiple groups, each with a target container
- **Subtask:** One raw `Subtask` with parallel per-object ladders; final conditions match termination destinations and release requirements
- **Attributes:** `sorting`, `semantics`; add colors/spatial descriptions only after inspecting the scene

```python
from dataclasses import dataclass

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.core.scenes.utils import import_scene
from functools import partial
from robolab.core.task.conditionals import object_groups_in_containers, object_grabbed, object_in_container
from robolab.core.task.subtask import Subtask
from robolab.core.task.task import Task

@configclass
class FoodPackingByColorTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_groups_in_containers,
        params={
            "groups": [
                {
                    "object": ["coffee_can"],
                    "container": "bin_a06",
                    "logical": "all",
                    "require_contact_with": False,
                    "require_gripper_detached": True,
                },
                {
                    "object": ["mustard", "sugar_box"],
                    "container": "bin_b03",
                    "logical": "all",
                    "require_contact_with": False,
                    "require_gripper_detached": True,
                },
            ]
        },
    )


@dataclass
class FoodPackingByColorTask(Task):
    contact_object_list = [
        "bin_a06", "bin_b03", "cheez_it", "chocolate_pudding",
        "coffee_can", "mustard", "sugar_box", "spam_can", "tomato_soup_can", "table",
    ]
    scene = import_scene("food_packing.usda", contact_object_list)
    terminations = FoodPackingByColorTerminations
    instruction = {
        "default": "Put coffee_can in bin_a06 and mustard and sugar_box in bin_b03",
        "vague": "Sort the items into their designated bins",
        "specific": "Place coffee_can in bin_a06 and mustard and sugar_box in bin_b03, in any order",
    }
    episode_length_s: int = 120
    attributes = ['sorting', 'semantics']
    subtasks = [
        Subtask(
            name="sort_objects",
            conditions={
                obj: [
                    (partial(object_grabbed, object=obj), 0.0),
                    (partial(object_in_container, object=obj, container=destination,
                             tolerance=0.01, require_contact_with=False,
                             require_gripper_detached=True), 1.0),
                ]
                for obj, destination in {
                    "coffee_can": "bin_a06", "mustard": "bin_b03", "sugar_box": "bin_b03",
                }.items()
            },
            logical="all", score=1.0,
        ),
    ]
```

The current `pick_and_place_grouped` implementation has a different final release
check; do not substitute it here without verifying equivalence in the current
source. Raw conditions keep all groups parallel and preserve the requested goal.

## Example 3: Ordered Stacking

**Goal:** Stack colored blocks in a specific order.

- **Conditional:** `released_stack` composes ordered `stacked` with release of every block
- **Subtask:** Raw `Subtask` with `partial(released_stack, ...)` tracks released incremental pairs
- **Attributes:** `stacking`, `color`
- **Note:** Subtask scores are split across pairs (0.33 + 0.33 + 0.34 = 1.0)

```python
from dataclasses import dataclass
from functools import partial

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.core.scenes.utils import import_scene
from robolab.core.task.conditionals import stacked, object_dropped
from robolab.core.task.subtask import Subtask
from robolab.core.task.task import Task

def released_stack(env, objects, order="bottom_to_top", env_id=None):
    result = stacked(env, objects=objects, order=order, env_id=env_id)
    for obj in objects:
        result = result & object_dropped(env, object=obj, env_id=env_id)
    return result


@configclass
class BlockStackingTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=released_stack,
        params={
            "objects": ["red_block", "blue_block", "green_block", "yellow_block"],
            "order": "bottom_to_top",
        },
    )


@dataclass
class BlockStackingSpecifiedOrderTask(Task):
    contact_object_list = ["red_block", "blue_block", "green_block", "yellow_block", "table"]
    scene = import_scene("colored_blocks.usda", contact_object_list)
    terminations = BlockStackingTerminations
    instruction = {
        "default": "Stack the blocks in the order from bottom to top: red, blue, green, yellow",
        "vague": "Stack in the order of red, blue, green, yellow",
        "specific": "Build a tower by placing the red block first, then the blue block on top, then the green, and finally the yellow block on top as a single tower",
    }
    episode_length_s: int = 90
    attributes = ['stacking', 'color']
    subtasks = [
        Subtask(
            conditions=partial(released_stack, objects=["red_block", "blue_block"], order="bottom_to_top"),
            score=0.33,
        ),
        Subtask(
            conditions=partial(released_stack, objects=["blue_block", "green_block"], order="bottom_to_top"),
            score=0.33,
        ),
        Subtask(
            conditions=partial(released_stack, objects=["green_block", "yellow_block"], order="bottom_to_top"),
            score=0.34,
        ),
    ]
```

`released_stack` adds release to the existing stack predicate; it does not add
sustained stability or penetration rejection. The underlying `stacked` check can
accept coincident overlapping blocks if contact is reported. Keep geometry and
physics validation separate, test representative invalid outcomes, and use a
stricter condition if the task requires those guarantees. This example is not a
certification of physical validity.

## Released, resting placement

For the Simple-PnP resting criterion, reuse the same dictionary for the termination
and final subtask check (choose the actual object/container names):

```python
from robolab.core.task.conditionals import object_resting_in_container, pick_and_place

placement_params = {
    "gripper_name": "gripper", "exclude_contact_with": "table",
    "settle_time_s": 0.5, "linear_threshold": 0.05, "angular_threshold": 0.5,
    "position_tolerance": 0.001, "rotation_tolerance_deg": 0.5,
}
success = DoneTerm(
    func=object_resting_in_container,
    params={"object": "plasticpail_a01", "container": "grey_bin", **placement_params},
)
subtasks = [pick_and_place(
    object=["plasticpail_a01"], container="grey_bin", logical="all", score=1.0,
    placement_func=object_resting_in_container, placement_params=placement_params,
)]
```

This checks final placement, not initial scene quality. Protrusion and stable
rim-spanning placements are allowed. Use a different explicit goal if the user
requires full enclosure or a particular target orientation.
