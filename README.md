# Uncertainty-Allocation-Tube-MPC
This repository is the official implementation for the paper: **"Uncertainty Allocation-based Tube Model Predictive Control for Building Energy Management"**.

Authors: Qi Li, Wenbo Zeng, Xueyuan Cui, Yi Wang

![Project Overview Figure](pipeline/pipieline.png)

This paper proposes a novel Uncertainty Allocation-based Tube MPC (UA-TMPC) framework. Departing from conventional static policies, we introduce a cost-aware active uncertainty allocation mechanism that treats allocation coefficients as decision variables, thereby dynamically directing forecasting errors towards the most cost-effective flexible resources. To address the resulting mathematical challenges of decision-dependent uncertainty, we construct adaptive ambiguity tubes that proactively adjust their geometry based on the allocation strategy. Furthermore, a soft-constrained formulation is integrated to mitigate the conservatism of hard tube bounds, enhancing solution feasibility without compromising system reliability.

# Environment
Python version: 3.8.17

The must-have packages can be installed by running
```bash
pip install -r requirements.txt
```
# Data
This project uses the data from multiple sources.
- **Download Link** You can download it from our google drive(https://drive.google.com/drive/folders/1EQyzTMvkdHomEoAbs5OwajAbKfOHnszL?usp=sharing)
- **Setup** After downloading, place all the files in the same directory as the code. The expected directory should be
  ```
  your-repository/
  ├── building data/
  │   └── xxxx.model/
  ├── 15temp_models/
  │   └── xxxx.model/
  ├── 15pv_models/
  │   └── xxxx.model/
  ├── 15load_models/
  │   └── xxxx.model/
  ├── United Kingdom.csv/
  ├── 22solar_data15.csv/
  ├── 23solar_data15.csv/
  ├── final_single.ipynb/
  ├── final_full.ipynb/
  ├── xxxxx.ipynb/
  └── README.md
  ```

This repository includes Jupyter Notebooks for data exploration, model demonstration, and result analysis.
- **'f'**: 
- **'f'**: 
- **'c'**: 
- **'d'**: 
- **'sb'**: 
- **'s'**: 
- **''**: 

