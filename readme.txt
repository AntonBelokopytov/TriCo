## Folder Layout

```text
/home/alex/2026_TriCo
├── assets
│   └── common_pics
├── code
│   ├── core
│   ├── real_data_examples
│   └── simulations
├── data
│   ├── external
│   └── support
├── literature
│   ├── graph_methods
│   ├── graph_methods_time_series
│   ├── source_separation
│   └── SPoC
└── readme.txt
```

## What Goes Where

- `literature/`: papers and reading materials.
- `code/core/`: core algorithms and reusable source code.
- `code/simulations/`: simulation scripts and related code.
- `code/real_data_examples/`: runnable examples on real datasets.
- `data/support/`: shared support data required by scripts.
- `data/external/`: external datasets grouped by task/example.
- `assets/common_pics/`: figures and images used across the project.

## Data Link

Site-packages and data:
https://drive.google.com/drive/u/0/folders/1uxSsBCCkTNMPzyf7fSFK3O87xk00QWFL

## Fieldtrip

It is better to import Fieldtrip by strings because it has some problems with uploading:

close all
clear
clc

ft_path = 'C:\Users\ansbel\Documents\2Git\fieldtrip\fieldtrip';

if ~exist('ft_defaults','file')
    addpath(ft_path);
end