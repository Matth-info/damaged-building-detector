# Downloading Data

This project provides custom PyTorch `DataLoader` implementations for several remote sensing and geospatial datasets commonly used in change detection, segmentation, and classification tasks.

In line with the principles of the [Model Openness Framework (MOF)](https://arxiv.org/pdf/2403.13784), this document outlines how to access each dataset to ensure transparency, reproducibility, and accessibility in AI research.

A selection of example images from each dataset is available in the data/data_samples directory.

---

# 📦 Datasets

# 1. **LEVIR-CD**
- **Description:** A benchmark dataset for change detection in high-resolution remote sensing images.
- **Source Paper:** [_A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection_](https://doi.org/10.3390/rs12101662)

---

# 2. **Open Cities AI Dataset**
- **Description:** Data for urban structure analysis and disaster risk management from multiple cities.
- **Source Page:** [Open Cities AI Challenge](https://source.coop/repositories/open-cities/ai-challenge/description)

---

# 3. **xBD Dataset**
- **Description:** A large-scale dataset for post-disaster damage assessment from satellite imagery (pre-disaster images are available).
- **Source Paper:** [_xBD: A Dataset for Assessing Building Damage from Satellite Imagery_](https://arxiv.org/abs/1911.09296)

---

# 4. **Cloud Detection (DrivenData)**
- **Description:** Sentinel-2 imagery for cloud cover segmentation tasks.
- **Source Page:** [Cloud Cover Detection Challenge](https://source.coop/repositories/radiantearth/cloud-cover-detection-challenge/description)

---

# 5. **Puerto Rico Dataset**
- **Description:**  Satellite imagery of Puerto Rico captured before and after Hurricane Maria, which struck in 2017. This dataset was released as part of the [2024 EY Open Science Data Challenge](https://challenge.ey.com/2024), and is useful for tasks such as disaster impact assessment and change detection.
- **Source:**
    ```shell
    !wget "https://challenge.ey.com/api/v1/storage/admin-files/Pre_Event_San_Juan.tif" -O "Pre_Event_San_Juan.tif"
    !wget "https://challenge.ey.com/api/v1/storage/admin-files/Post_Event_San_Juan.tif" -O "Post_Event_San_Juan.tif"
    ```
---

# 📘 Notes
- Some datasets may require manual download or license agreement.
- Ensure you follow the usage terms defined by the original data providers.

---

# 📂 Using the Dataloaders
Refer to the [`datasets/`](./datasets) directory for usage examples and integration with training pipelines.
