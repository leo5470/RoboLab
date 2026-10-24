# Simple pick-and-place benchmark

6,754 tasks and matching scenes across 136 catalog entries. Each task grasps one tabletop item and places it into `grey_bin`, with a 50-second episode limit. Success requires a released object supported by an upright, stationary bin, with no table contact. Both object and bin must remain below 0.05 m/s linear speed and 0.5 rad/s angular speed, with poses staying within 1 mm and 0.5 degrees of their initial settled poses for 0.5 simulated seconds. Protrusion and stable rim-spanning placements are accepted. Termination and full subtask credit use the same condition.

## Filtering

The original 7,200 layouts (50 per catalog entry) were filtered using `output/pi05_simple_pnp_5eps_combined_launch_filtered`. All 446 excluded tasks and matching scenes are absent from this directory and the scene manifest. Exclusions cover launch-audit falls, replaced asset scales, and the explicitly excluded objects; geometry warnings alone do not exclude tasks.

`exclusions.json` records every excluded task, its reason, and source hashes. Original task IDs and layout suffixes are preserved, so some objects have fewer than 50 layouts. Count catalog entries by index: names are not necessarily unique.

## Use

Run from the RoboLab repository root with an already-running pi05 server:

```bash
.venv/bin/python policies/pi0_family/run.py \
    --policy pi05 --headless --device cuda:0 --remote-port 8000 \
    --task-dirs single_pnp --task SimplePnp000BlueBlock00Task \
    --num-envs 1 --num-runs 1 --video-mode none \
    --output-folder-name simple_pnp_smoke_test
```

Tasks register automatically at startup. Omit `--task` to select the whole set; use explicit task batches to limit registration time and memory. No additional exclusion filtering is needed for this committed task set.

Run `git lfs pull` after cloning to obtain the official assets and the five repaired models in `assets/objects/repaired/`. Small generated scene layers are stored as ordinary text files. Existing RoboLab/IsaacLab dependencies and the policy server must be installed separately. Repaired models use relative texture paths so the repository can be installed at any location.

Open `assets/scenes/simple_pnp/layout_viewer.html` to inspect the retained layouts without simulation. Both manifests describe only the retained set.

## Retained layouts

| Catalog index | Object | Layouts |
| --- | --- | ---: |
| 000 | blue_block | 50 |
| 001 | green_block | 49 |
| 002 | red_block | 50 |
| 003 | yellow_block | 50 |
| 004 | avocado01 | 50 |
| 005 | lemon_01 | 50 |
| 006 | lemon_02 | 49 |
| 007 | lime01 | 50 |
| 008 | lychee01 | 50 |
| 009 | orange_01 | 50 |
| 010 | orange_02 | 50 |
| 011 | pomegranate01 | 50 |
| 012 | red_onion | 50 |
| 023 | measuring_cups | 50 |
| 024 | measuring_cups_1 | 50 |
| 025 | measuring_spoon | 50 |
| 029 | spoon | 50 |
| 030 | spoon_1 | 50 |
| 031 | spoon_2 | 50 |
| 032 | bbq_sauce_bottle | 50 |
| 033 | butter | 50 |
| 034 | canned_mushrooms | 50 |
| 035 | canned_peaches | 50 |
| 036 | canned_tuna | 50 |
| 037 | chocolate_pudding_mix | 50 |
| 038 | corn_can | 50 |
| 039 | cream_cheese | 50 |
| 040 | granola_bars | 50 |
| 041 | green_beans_can | 50 |
| 042 | ketchup_bottle | 50 |
| 043 | macaroni_and_cheese | 50 |
| 044 | mayonnaise_bottle | 50 |
| 045 | milk_carton | 49 |
| 046 | mustard_bottle | 49 |
| 047 | oatmeal_raisin_cookies | 50 |
| 048 | orange_juice_carton | 50 |
| 049 | parmesan_cheese_canister | 50 |
| 050 | peas_and_carrots | 50 |
| 051 | pineapple_slices_can | 50 |
| 052 | pitted_cherries | 50 |
| 053 | popcorn_box | 50 |
| 054 | raisin_box | 50 |
| 055 | ranch_dressing | 50 |
| 056 | spaghetti | 50 |
| 057 | tomato_sauce_can | 50 |
| 058 | yogurt_cup | 50 |
| 059 | bbq_sauce_bottle | 50 |
| 061 | ceramic_mug | 50 |
| 063 | coffee_pot | 50 |
| 064 | computer_mouse | 30 |
| 066 | foam_roller | 50 |
| 067 | frozen_vegetable_block | 50 |
| 068 | frozen_waffles | 50 |
| 069 | glasses | 50 |
| 071 | lizard_figurine | 50 |
| 073 | mug | 49 |
| 074 | pitcher | 50 |
| 075 | potato_masher | 49 |
| 076 | remote_control | 50 |
| 077 | rubiks_cube | 50 |
| 078 | smartphone | 50 |
| 079 | soup_can | 50 |
| 081 | storage_box | 49 |
| 082 | wooden_bowl | 50 |
| 085 | apple | 49 |
| 086 | bagel_00 | 50 |
| 087 | bagel | 50 |
| 090 | gregorys_coffee_cup | 47 |
| 091 | lunchbag | 49 |
| 092 | red_bell_pepper | 50 |
| 093 | snickers_bar | 50 |
| 111 | blackandbrassbowl_small | 50 |
| 217 | crabbypenholder | 50 |
| 220 | fork_big | 50 |
| 221 | fork_small | 50 |
| 230 | milkjug_a01 | 50 |
| 231 | milkjug_a02 | 49 |
| 232 | milkjug_a03 | 50 |
| 234 | naturalbostonroundbottle_a01 | 50 |
| 235 | naturalbostonroundbottle_a02 | 50 |
| 236 | naturalbostonroundbottle_a03 | 50 |
| 237 | plasticjerrican_a | 50 |
| 238 | plasticjerrican_a01 | 50 |
| 239 | plasticjerrican_a02 | 50 |
| 242 | plasticpail_a01 | 50 |
| 243 | plasticpail_a02 | 50 |
| 247 | plate_small | 50 |
| 248 | pumpkinlarge | 50 |
| 249 | pumpkinsmall | 50 |
| 251 | screwtoppail_a01 | 50 |
| 256 | serving_bowl | 50 |
| 258 | spatula_01 | 50 |
| 260 | spatula_03 | 50 |
| 261 | spatula_04 | 50 |
| 262 | spatula_05 | 50 |
| 263 | spatula_06 | 50 |
| 264 | spatula_07 | 50 |
| 265 | spatula_08 | 50 |
| 266 | spatula_09 | 50 |
| 267 | spatula_10 | 50 |
| 268 | spatula_11 | 50 |
| 269 | spatula_12 | 50 |
| 270 | spatula_13 | 50 |
| 271 | spatula_14 | 50 |
| 272 | spatula_15 | 50 |
| 273 | spatula_16 | 50 |
| 274 | spatula_17 | 50 |
| 275 | spatula_18 | 50 |
| 276 | spatula_19 | 50 |
| 277 | spoon_big | 50 |
| 278 | spoon_small | 50 |
| 279 | squarepail_a01 | 50 |
| 283 | utilityjug_a01 | 47 |
| 284 | utilityjug_a02 | 50 |
| 286 | whitepackerbottle_a01 | 50 |
| 287 | whitepackerbottle_a02 | 50 |
| 288 | whitepackerbottle_a03 | 50 |
| 290 | banana | 50 |
| 291 | bowl | 50 |
| 292 | bowl | 50 |
| 294 | cheez_it | 50 |
| 295 | chocolate_pudding | 50 |
| 296 | clamp | 50 |
| 298 | cordless_drill | 50 |
| 299 | dry_erase_marker | 50 |
| 300 | jello | 50 |
| 301 | mug | 50 |
| 302 | mustard | 49 |
| 303 | pitcher | 50 |
| 304 | scissors | 50 |
| 305 | soft_scrub | 49 |
| 306 | spam_can | 49 |
| 307 | spring_clamp | 50 |
| 308 | sugar_box | 50 |
| 309 | tomato_soup_can | 43 |
| 310 | tuna_can | 50 |
