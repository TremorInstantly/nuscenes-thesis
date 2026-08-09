# nuScenes Dataset Setup

This directory is used as the local data root for the nuScenes dataset required by this project.

The dataset itself is not included in this repository. Users must download it separately from the official nuScenes website.

## 1. Download nuScenes

Download the required dataset release from the official nuScenes download page:

https://www.nuscenes.org/download

A nuScenes account and acceptance of the dataset terms are required to download the data. The dataset is provided for non-commercial use under the applicable nuScenes terms.

For the default configuration in this repository, use:

v1.0-mini


The official nuScenes tutorial also demonstrates downloading and extracting the `v1.0-mini` release.

---

## 2. Extract the Dataset

After downloading the "v1.0-mini" archive, extract it so that the directory structure is approximately:


data/
└── nuscenes_mini/
    ├── samples/
    ├── sweeps/
    ├── maps/
    ├── v1.0-mini/
    └── ...

The exact contents may vary depending on the additional nuScenes expansions that have been downloaded.

The important requirement is that "config.py" points "DATAROOT" to the directory containing the nuScenes dataset:

DATAROOT = ROOT_PATH / "data/nuscenes_mini"

The nuScenes devkit should then be able to initialize the dataset using:


from nuscenes.nuscenes import NuScenes
nusc = NuScenes(
    version="v1.0-mini",
    dataroot=DATAROOT
)


The official tutorial uses the same "v1.0-mini" version and a dataset root containing the extracted nuScenes files.

---

## 3. Map Expansion

This project uses map information during preprocessing, particularly for extracting lane-related information.

Therefore, the required nuScenes Map Expansion must also be downloaded and installed in the same nuScenes data root.

The official nuScenes site provides the map and other dataset expansions through the download page:

https://www.nuscenes.org/download

After extraction, the map files should be available under the dataset's `maps/` directory.

The nuScenes prediction tutorial demonstrates the map API and shows how lane information such as lane records, incoming lanes, outgoing lanes, and lane centerline information can be accessed through the map expansion.

For example:

from nuscenes.map_expansion.map_api import NuScenesMap
nusc_map = NuScenesMap(
    map_name="singapore-onenorth",
    dataroot=DATAROOT
)

IMPORTANT: The map expansion is required for the preprocessing pipeline used by this project. Installing only the basic dataset may therefore be insufficient.

---

## 4. Using `v1.0-trainval`

The project can also be used with the full train/validation release instead of the mini dataset.

Download the appropriate v1.0-trainval release from the official nuScenes website and extract it into a separate directory.

For example:

data/
├── nuscenes_mini/
│   ├── samples/
│   ├── sweeps/
│   ├── maps/
│   └── v1.0-mini/
│
└── nuscenes_trainval/
    ├── samples/
    ├── sweeps/
    ├── maps/
    └── v1.0-trainval/

Then change the dataset configuration accordingly:

DATAROOT = ROOT_PATH / "data" / "nuscenes_trainval"
VERSION = "v1.0-trainval"

The same map expansion should be available within the corresponding dataset root if the preprocessing pipeline requires map information.

