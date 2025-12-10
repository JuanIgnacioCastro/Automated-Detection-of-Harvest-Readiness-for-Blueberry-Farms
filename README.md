# Automated-Detection-of-Harvest-Readiness-for-Blueberry-Farms
FIU Data Science &amp; Artificial Intelligence capstone project: automated computer-vision pipeline (YOLOv8 + Python) to detect ripe vs. green blueberries, estimate harvest-readiness at plant/acre level, and support data-driven harvest timing decisions for commercial blueberry farms.

# Automated Detection of Harvest Readiness for Blueberry Farms

Juan Ignacio Castro  
Data Science and Artificial Intelligence – Business Analytics Track  
Florida International University (FIU) – IDC 6940 Capstone in Data Science  

---

## 1. Project Overview

This repository contains the code for my FIU master’s capstone project, **“Automated Detection of Harvest Readiness for Blueberry Farms.”**  

The goal of the project is to combine:

- **Computer vision** (ripe vs. green blueberry detection),
- **Biological ripening models** (fruit development over time), and
- **Economic modeling** (manual vs. machine harvest profitability)

to support **data-driven harvest timing decisions** for commercial blueberry growers.

The codebase is organized into four main branches that correspond to the core components of the system:

1. `COMPUTER-VISION` – YOLOv8 object detection for ripe/green berries  
2. `SYNTHETHIC-FARM` – synthetic farm generation and plant-level simulations  
3. `ECONOMIC-AND-BIOLOGICAL-MODEL` – ripening curves + harvest economics  
4. `CHAT-BOT-FRUIT` – conversational interface that explains results to users  

All branches contain **working code** used in the empirical experiments for the capstone.

---

## 2. Branches and Components

### 2.1 `COMPUTER-VISION`

Computer-vision pipeline to detect **ripe vs. green blueberries** on plants and estimate harvest-ready fruit mass.

Main ideas:

- YOLOv8 object detection model trained on **two classes**  
  - `0` = green / unripe berry  
  - `1` = ripe / harvestable berry
- Inference scripts to:
  - Run detection on plant or row images
  - Count berries by class
  - Aggregate results at plant, acre, and farm level for downstream modeling

**Image data**

- Training/validation/test images were collected and labeled using **Roboflow**.
- **Images are *not* included in this repository** due to size and licensing considerations.
- I can share the exact datasets upon request, but for reproducibility I recommend:
  1. Creating your own Roboflow project,
  2. Uploading blueberry images,
  3. Using similar two-class labels (green vs. ripe),
  4. Exporting data in YOLOv8 format and placing it in the expected directory structure.

---

### 2.2 `SYNTHETHIC-FARM`

Code for building a **synthetic blueberry farm** with thousands of plants, each having:

- Total berry counts
- Ripe vs. green proportions
- Acre and plant IDs
- Day-by-day evolution based on biological ripening curves

Key elements:

- Generation of ~12,000+ unique plants (multi-acre farm)
- Use of probability distributions to vary:
  - Total berries per plant
  - Ripe ratios
  - Within-acre and across-acre variability
- Output: clean tabular datasets that feed into both the **ripening model** and the **economic model**.

This branch essentially creates a “virtual farm” that mimics realistic variation while remaining fully controlled and reproducible for experimentation.

---

### 2.3 `ECONOMIC-AND-BIOLOGICAL-MODEL`

This branch integrates:

1. **Biological ripening model**
   - Models how berries transition from green → ripe over time.
   - Uses parametric curves to approximate:
     - Start of ripening,
     - Peak harvest window,
     - Decline in quality or quantity afterwards.
   - Produces **daily estimates of ripe kilograms per acre/plant**.

2. **Economic harvest model**
   - Compares **manual picking** vs. **machine harvest**, including:
     - Picker productivity (kg/hour)
     - Wages and hours per day
     - Minimum/maximum pickers
     - Packaging/material costs per kg
   - Machine harvest:
     - Fixed daily cost
     - Throughput (kg/hour)
     - Penalty for destroyed green berries (green-loss fraction)
   - Market price curves over the harvest season (non-linear price drops).

3. **Decision logic**
   - For each simulated day, the model evaluates:
     - Revenue from harvesting now vs. waiting
     - Net profit for manual vs. machine harvest
   - Produces a **harvest decision schedule** and **profit curves**.

Outputs include pandas DataFrames and plots used in the capstone report to analyze harvest strategies under different scenarios.

---

### 2.4 `CHAT-BOT-FRUIT`

Conversational interface that allows a non-technical user (e.g., a grower or farm manager) to query:

- Current simulated farm status
- Recommended harvest days
- Expected yields and revenues
- Model assumptions and parameters

The chatbot uses **Grok** (xAI) as the LLM backend.

> ⚠️ **API Key Required**  
> To run this branch you must:
> 1. Obtain a **Grok API key** from xAI.  
> 2. Store it in your environment (e.g., `.env` file or environment variable like `GROK_API_KEY`).  
> 3. Update the configuration file or notebook to read the key from your environment.

No external keys are committed to this repository.

---

## 3. Environment & Dependencies

The project is primarily written for **Python** and **Jupyter/Google Colab** workflows.

Typical stack (exact versions may differ by branch):

- Python 3.x  
- `numpy`, `pandas`, `matplotlib`, `seaborn` (for data & plots)  
- `scipy` (for distributions and curve modeling)  
- `ultralytics` (YOLOv8)  
- `opencv-python` (image preprocessing)  
- `tqdm` (progress bars)  
- `python-dotenv` or similar (for managing API keys)  

Please consult the specific branch for `requirements.txt` or environment instructions.  
Most notebooks were originally executed in **Google Colab**, so they should run with minimal setup once dependencies are installed.

---

## 4. How to Use This Repository

### 4.1 Cloning

```bash
git clone https://github.com/JuanIgnacioCastro/Automated-Detection-of-Harvest-Readiness-for-Blueberry-Farms.git
cd Automated-Detection-of-Harvest-Readiness-for-Blueberry-Farms


