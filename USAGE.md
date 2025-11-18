# Quick Start Guide

## Prerequisites
- Python 3.9 or higher
- pip package manager
- Jupyter Notebook

## Installation Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/dl1413/TextbookBiasDetection.git
   cd TextbookBiasDetection
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Linux/Mac:
   source venv/bin/activate
   
   # On Windows:
   venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API keys** (optional, for real LLM analysis)
   ```bash
   cp .env.example .env
   # Edit .env and add your actual API keys
   ```

5. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

6. **Open the notebook**
   - Navigate to `textbook_bias_detection.ipynb`
   - Run cells sequentially (Cell → Run All)

## What the Notebook Does

### Phase 1: Data Setup
- Generates synthetic dataset (4,500 passages from 150 textbooks)
- In production: Load real textbook passages and LLM ratings

### Phase 2: Reliability Analysis
- Calculates Krippendorff's Alpha for LLM ensemble
- Validates inter-rater reliability across dimensions

### Phase 3: Factor Analysis
- Performs EFA with varimax rotation
- Identifies latent bias dimensions
- Calculates factor scores

### Phase 4: Bayesian Modeling
- Fits hierarchical models using PyMC
- Estimates publisher-type effects
- Quantifies uncertainty with MCMC

### Phase 5: Visualization & Reporting
- Generates comprehensive plots
- Produces statistical summaries
- Saves results to files

## Output Files

After running the notebook, you'll find:

- `results/bayesian_results.csv` - Publisher effect estimates
- `results/analysis_summary.csv` - Overall statistics
- `figures/inter_rater_reliability.png` - Reliability plot
- `figures/scree_plot.png` - Factor analysis scree plot
- `figures/factor_loadings_heatmap.png` - Factor loadings
- `figures/factor_scores_by_publisher.png` - Box plots
- `figures/trace_commercial_influence.png` - MCMC diagnostics
- `figures/forest_commercial_influence.png` - Effect sizes
- `figures/comprehensive_forest_plot.png` - All factors

## Customization

### Using Real Data

Replace the `generate_synthetic_dataset()` function with code to:
1. Load textbook passages from your corpus
2. Call LLM APIs for ratings
3. Format data into the expected structure

### Adjusting Model Parameters

- **Number of factors**: Change `n_factors` in Section 6
- **MCMC parameters**: Adjust `sample()` parameters in Section 8
- **Prior distributions**: Modify priors in PyMC model specification

### Adding Disciplines or Publishers

Update the constants in Section 2:
```python
PUBLISHER_TYPES = ['For-Profit', 'University Press', 'Open-Source', 'Your-Type']
DISCIPLINES = ['Biology', 'Chemistry', ..., 'Your-Discipline']
```

## Troubleshooting

**Issue**: Import errors
- **Solution**: Ensure all dependencies are installed: `pip install -r requirements.txt`

**Issue**: PyMC sampling errors
- **Solution**: Try increasing `tune` parameter or reducing `target_accept`

**Issue**: Memory errors with large datasets
- **Solution**: Process data in batches or use a machine with more RAM

**Issue**: Notebook cells fail to run
- **Solution**: Restart kernel and run cells sequentially from the top

## Support

For questions or issues:
- Open an issue on GitHub
- Check the PDF publication for theoretical background
- Review the inline documentation in notebook cells

## Citation

If you use this notebook in your research:

```bibtex
@mastersthesis{lankeaux2025textbook,
  author = {Lankeaux, Derek},
  title = {Detecting Publisher Bias in Academic Textbooks Using Bayesian Ensemble Methods and Large Language Models},
  school = {Rochester Institute of Technology},
  year = {2025},
  type = {MS Capstone Project}
}
```
