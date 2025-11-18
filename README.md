# Textbook Bias Detection

**Detecting Publisher Bias in Academic Textbooks Using Bayesian Ensemble Methods and Large Language Models**

A comprehensive statistical framework for quantifying editorial influence in educational materials.

## Overview

This project implements the methodology described in the research publication "Detecting Publisher Bias in Academic Textbooks Using Bayesian Ensemble Methods and Large Language Models" by Derek Lankeaux (RIT MS Applied Statistics Capstone, November 2025).

The analysis combines:
- **LLM Ensemble Rating System** - Multi-model assessment (GPT-4, Claude-3, Llama-3)
- **Exploratory Factor Analysis** - Uncovering latent bias dimensions
- **Bayesian Hierarchical Modeling** - Quantifying publisher-type effects with PyMC
- **Comprehensive Validation** - Inter-rater reliability and validity testing

## Key Features

✨ **Multi-dimensional Bias Assessment**: Evaluates textbooks across 5 theoretical dimensions:
- Perspective Balance
- Source Authority
- Commercial Framing
- Certainty Language
- Ideological Framing

📊 **Rigorous Statistical Framework**:
- Exploratory Factor Analysis with varimax rotation
- Bayesian hierarchical models with full uncertainty quantification
- MCMC sampling with convergence diagnostics
- Effect size estimation and credible intervals

🔬 **Validation Studies**:
- Krippendorff's Alpha for inter-rater reliability
- Convergent validity with expert coders
- Discriminant validity tests

## Installation

### Prerequisites
- Python 3.9+
- Jupyter Notebook

### Setup

1. Clone the repository:
```bash
git clone https://github.com/dl1413/TextbookBiasDetection.git
cd TextbookBiasDetection
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Analysis

1. Launch Jupyter Notebook:
```bash
jupyter notebook
```

2. Open `textbook_bias_detection.ipynb`

3. Run all cells sequentially to:
   - Generate synthetic dataset (or load real data)
   - Calculate inter-rater reliability
   - Perform factor analysis
   - Fit Bayesian hierarchical models
   - Generate visualizations and reports

### Notebook Structure

The notebook is organized into 11 main sections:

1. **Setup and Imports** - Load required libraries
2. **Configuration** - Define constants and parameters
3. **Data Generation** - Simulated dataset (replace with real data)
4. **Data Preprocessing** - Calculate consensus ratings
5. **Inter-Rater Reliability** - Krippendorff's Alpha analysis
6. **Exploratory Factor Analysis** - EFA with varimax rotation
7. **Descriptive Statistics** - Publisher type comparisons
8. **Bayesian Hierarchical Models** - PyMC implementation
9. **Comprehensive Visualization** - Forest plots and dashboards
10. **Key Findings** - Summary of results
11. **Conclusions** - Implications and future directions

## Research Questions

**RQ1**: What latent dimensions underlie systematic variation in textbook content presentation?

**RQ2**: Do for-profit, university press, and open-source publishers exhibit systematically different patterns?

**RQ3**: Do publisher-type effects vary across academic disciplines?

**RQ4**: Do LLM ensemble ratings demonstrate acceptable validity and reliability?

## Key Findings (from simulated data)

- ✅ LLM ensemble achieves excellent inter-rater reliability (α > 0.80)
- ✅ Four-factor structure explains >80% of variance
- ✅ For-profit publishers show higher commercial influence
- ✅ Open-source materials exhibit greater perspective diversity
- ✅ Effects are educationally meaningful (Cohen's d > 0.5)

## Dataset

The research design calls for:
- **150 textbooks** across 3 publisher types (For-Profit, University Press, Open-Source)
- **6 disciplines** (Biology, Chemistry, Computer Science, Economics, Psychology, History)
- **4,500 passages** (30 per textbook)
- **15-dimensional ratings** (5 dimensions × 3 LLM models)

*Note: The current notebook uses simulated data for demonstration. Real implementation requires an actual textbook corpus and LLM API access.*

## Files

- `textbook_bias_detection.ipynb` - Main analysis notebook
- `requirements.txt` - Python dependencies
- `Detecting Publisher Bias in Academic Textbooks Using Bayesian Ensemble Methods and Large Language ModelsF.pdf` - Full research publication
- `data/` - Data directory (create for real datasets)
- `results/` - Analysis results (auto-generated)
- `figures/` - Visualizations (auto-generated)

## Citation

If you use this methodology in your research, please cite:

```
Lankeaux, D. (2025). Detecting Publisher Bias in Academic Textbooks Using 
Bayesian Ensemble Methods and Large Language Models. MS Applied Statistics 
Capstone Project, Rochester Institute of Technology.
```

## License

MIT License - See LICENSE file for details

## Contact

Derek Lankeaux - Rochester Institute of Technology

For questions or collaboration opportunities, please open an issue on GitHub.

## Acknowledgments

This project was developed as part of the MS Applied Statistics program at RIT's School of Mathematical Sciences.
