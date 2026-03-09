# Uncertainty-Allocation-Tube-MPC
This repository is the official implementation for the paper: **"Uncertainty Allocation-based Tube Model Predictive Control for building energy management"**.

![Project Overview Figure](pipeline/pipieline.png)


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
- **'final_single.ipynb'**: Provides a detailed simulation of UA-TMPC, H-UA-TMPC, DMPC, SMPC, and RMPC for one single day.
- **'final_full.ipynb'**: Provides a simulation of the above methods for one complete month.
- **'comparison_result.ipynb'**: Provides a comparison between UA-TMPC, DMPC, SMPC, and RMPC.
- **'dynamic_behavior.ipynb'**: Plots the dynamic behavior comparison of UA-TMPC, DMPC, SMPC, and RMPC.
- **'scenario_test.ipynb'**: Abalation Studies of soft-formulation mechanism on different scaling factors
- **'sensitivity_analysis.ipynb'**: Sensitivity Analysis on weights of soft-mechanism
- **'size_sensitivity.ipynb'**: Sensitivity Analysis on scenario size
