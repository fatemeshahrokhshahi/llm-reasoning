## Enhanced Rule-Based Reasoning in Large Language Models: A Chain of Logic Approach for Conditional and Modal Inference

**Authors:** Fatemeh Shahrokhshahi¹, Farzan Mohammadi², Hayder Mohammedqasim³, and Ferdi Sonmez⁴

¹ fatemehshahrokhshahi@stu.aydin.edu.tr  
² farzan@blackscale.media  
³ hmohammedqasim@aydin.edu.tr  
⁴ ferdisonmez@aydin.edu.tr

### Abstract

This repository contains the experimental implementation and comprehensive analysis for evaluating the Chain of Logic prompting method on conditional and modal reasoning tasks. After conducting extensive research on rule-based reasoning approaches and identifying limitations in existing methods, we discovered the Chain of Logic methodology introduced by Servantez et al. (2024) for legal reasoning. Since no implementation was available, we developed a complete implementation from scratch based on their paper description and extended it to conditional and modal reasoning tasks. We then systematically evaluated this implementation using the experimental framework originally established by Holliday et al. (2024) across 16 logical inference patterns from the conditional and modal reasoning benchmark.

### Key Findings

Our evaluation demonstrates that Chain of Logic achieves an average accuracy of 80.84% across all models and patterns, representing substantial improvements over existing prompting methods:

- **Chain of Logic**: 80.84%
- **Chain of Thought**: 59.40% (+21.44 percentage point improvement)
- **Few-Shot**: 53.21% (+27.63 percentage point improvement)
- **Zero-Shot**: 45.71% (+35.13 percentage point improvement)

### Repository Structure

```
llm-reasoning/
├── Analysis.py                          # Main analysis script for all 4 prompting methods
├── COL-response-generator.py           # Chain of Logic response generation script
├── prompts/                            # Logical inference pattern prompts
│   ├── AC.json                        # Affirming Consequent prompts (20 items)
│   ├── DSmu.json                      # Disjunctive Syllogism must prompts (20 items)
│   ├── MP.json                        # Modus Ponens prompts (20 items)
│   └── ...                            # 13 more rule files (16 total)
├── data/                              # Model response data
│   ├── chain_of_logic/               # Chain of Logic method responses
│   │   ├── meta_llama_Meta_Llama_3_1_8B_Instruct_Turbo/
│   │   │   ├── AC/                   # AC_1_0.json to AC_20_0.json
│   │   │   ├── DSmu/                 # DSmu_1_0.json to DSmu_20_0.json
│   │   │   └── ...                   # 14 more rule directories (16 total)
│   │   ├── meta_llama_Meta_Llama_3_1_70B_Instruct_Turbo/
│   │   ├── meta_llama_Meta_Llama_3_1_405B_Instruct_Turbo/
│   │   └── ...
│   ├── chain_of_thought/             # Chain of Thought method responses
│   ├── few_shot/                     # Few-shot method responses
│   ├── zero_shot/                    # Zero-shot method responses
│   └── processed_results/            # Processed and analyzed data
├── results/                          # Experimental results and analysis
├── graphs/                           # Generated visualizations
├── requirements.txt                  # Python dependencies
└── config/                          # Experimental configuration files
```

### Experimental Framework

This work builds upon the experimental framework established by Holliday et al. (2024) for evaluating conditional and modal reasoning in large language models. Through extensive research on rule-based reasoning approaches, we identified the Chain of Logic methodology introduced by Servantez et al. (2024) for legal reasoning tasks. Since the original paper provided only the theoretical framework without any available implementation, we developed a complete implementation from scratch based on their methodology description. We then extended this implementation to the domain of conditional and modal reasoning, comparing it against established baseline approaches.

#### Models Evaluated
- Llama 3.1 Instruct 8B
- Llama 3.1 Instruct 70B  
- Llama 3.1 Instruct 405B

#### Logical Inference Patterns (16 total)
**Valid Rules:**
- MP (Modus Ponens)
- MT (Modus Tollens)
- NSFC (Narrow-Scope Free Choice)
- WSFC (Wide-Scope Free Choice)

**Invalid Rules (Logical Fallacies):**
- AC (Affirming the Consequent)
- DA (Denying the Antecedent)
- CONV (Conversion)
- INV (Inversion)
- AS (Antecedent Strengthening)
- CT (Contraposition with negation)
- CMP (Complex Modus Ponens)

**Modal Logic Rules:**
- DSmu (Disjunctive Syllogism with 'must')
- DSmi (Disjunctive Syllogism with 'might')
- MTmu (Modus Tollens with 'must')
- MTmi (Modus Tollens with 'might')
- MuDistOr (Must distributes over Or)
- MiAg (Might Agglomeration)

#### Prompting Methods Compared
- **Chain of Logic**: Six-step structured decomposition method (Servantez et al., 2024)
- **Chain of Thought**: Standard intermediate reasoning approach
- **Few-Shot**: Example-based prompting
- **Zero-Shot**: Direct prompting without examples

### Installation and Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/fatemeshahrokhshahi/llm-reasoning.git
   cd llm-reasoning
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure directory paths:**
   
   **For Analysis.py** - Update the directory paths in the script:
   ```python
   # Directory paths
   data_dir = r"C:\your\path\to\data"  # Path to your data directory
   output_dir = r"C:\your\path\to\output"  # Path where results will be saved
   ```

   **For COL-response-generator.py** - Update the paths in the `main()` function:
   ```python
   def main():
       base_path = r"C:\your\project\path"  # Base path for your project
       prompts_base_path = r"C:\your\path\to\prompts"  # Path to JSON prompt files
   ```

4. **Set up API keys (for generating new responses):**
   Edit the API key variables in `COL-response-generator.py`:
   ```python
   OPENAI_API_KEY = 'your_openai_key_here'
   ANTHROPIC_API_KEY = 'your_anthropic_key_here'
   GOOGLE_API_KEY = 'your_google_key_here'
   TOGETHER_API_KEY = 'your_together_key_here'
   MISTRAL_API_KEY = 'your_mistral_key_here'
   ```

### Usage

#### Analyzing Existing Results

To analyze the experimental results across all 4 prompting methods:

```bash
python Analysis.py
```

**What Analysis.py does:**
- Loads response data from all prompting methods (chain_of_logic, chain_of_thought, few_shot, zero_shot)
- Analyzes performance across 16 logical inference rules
- Generates comprehensive statistics and visualizations
- Performs statistical significance testing
- Creates publication-ready plots and detailed reports

**Generated outputs:**
- `detailed_results.csv` - Complete response data with correctness annotations
- `summary_statistics.csv` - Aggregated performance metrics
- `condition_performance.csv` - Performance by prompting method
- `statistical_comparisons.csv` - Statistical significance test results
- `prompting_method_analysis_report.txt` - Comprehensive analysis report
- Multiple visualization files (PNG format)

#### Generating New Chain of Logic Responses

To create new responses using the Chain of Logic method:

```bash
python COL-response-generator.py
```

**What COL-response-generator.py does:**
- Loads logical reasoning prompts from JSON files in the prompts directory
- Implements the Chain of Logic methodology through specialized 6-step system prompts for each logical rule
- Creates responses by calling various LLM APIs (OpenAI, Anthropic, Google, Together, Mistral)
- Applies rule-specific Chain of Logic reasoning frameworks to generate structured logical responses
- Saves Chain of Logic responses in the proper directory structure for later analysis by Analysis.py
- Includes rate limiting and comprehensive error handling for API calls

**Configuration:**
- Update `base_path` and `prompts_base_path` in the `main()` function
- Select which models to test by modifying the `get_available_models()` function
- Choose specific logical rules by editing the `rules_to_test` list

### Data Organization

Response data is organized hierarchically by prompting method, model, and logical rule type:

```
data/[method]/[model]/[rule]/[rule]_[number]_0.json
```

**Example:** `data/chain_of_logic/meta_llama_Meta_Llama_3_1_8B_Instruct_Turbo/AC/AC_1_0.json`

Each JSON file contains:
```json
{
  "user_prompt": "...",
  "system_prompt": "...",
  "model": "...",
  "temperature": 0,
  "responses": [
    {
      "content": "..."
    }
  ]
}
```

### Chain of Logic Methodology

The Chain of Logic method implements a structured 6-step reasoning process originally introduced by Servantez et al. (2024) for legal reasoning tasks. Since no implementation was available from the original paper, we developed our own complete implementation based on their methodology description and adapted it specifically for conditional and modal logical reasoning:

1. **Structured Input** - Identify premises and question clearly
2. **Rule Decomposition** - Break down logical components
3. **Logical Expression** - Formalize the logical structure
4. **Question Answering** - Apply logical analysis
5. **Element Recomposition** - Combine logical elements
6. **Resolve Expression** - Reach final conclusion

Each logical rule has a specialized system prompt that guides the model through this process while incorporating rule-specific logical principles. Our implementation extends the original legal domain approach to fundamental logical reasoning patterns including conditional statements, modal operators (necessity and possibility), and their combinations.

### Experimental Results

The Chain of Logic method demonstrates consistent improvements across all model sizes and logical rule types. Key findings:

- **Significant improvements** over all baseline methods (p < 0.05)
- **Largest gains** for complex modal reasoning tasks
- **Consistent performance** across different model sizes
- **Particular effectiveness** for logical fallacy detection

Detailed results and statistical analysis can be reproduced by running `Analysis.py` on the provided dataset.

### Requirements

```
pandas>=1.3.0
matplotlib>=3.3.0
seaborn>=0.11.0
numpy>=1.21.0
scipy>=1.7.0
tqdm>=4.62.0
openai>=1.0.0
anthropic>=0.3.0
google-generativeai>=0.3.0
together>=0.2.0
mistralai>=0.1.0
```

### Citation

If you use this data or build upon this work, please cite:

```bibtex
@article{llm_reasoning_2025,
  title={Enhancing Rule-Based Reasoning in Large Language Models: A Chain of Logic Approach for Conditional and Modal Inference},
  author={Shahrokhshahi, Fatemeh and Mohammadi, Farzan and Mohammedqasim, Hayder and Sonmez, Ferdi},
  year={2025},
  note={Submitted for publication},
  url={https://github.com/fatemeshahrokhshahi/llm-reasoning}
}
```

### Related Work

This work extends the experimental framework from:
- Holliday, W.H., Mandelkern, M., Zhang, C.E. "Conditional and Modal Reasoning in Large Language Models." ACL 2024. [Repository](https://github.com/wesholliday/llm-logic)

The Chain of Logic prompting method was originally introduced by:
- Servantez, S., Barrow, J., Hammond, K., Jain, R. "Chain of Logic: Rule-Based Reasoning with Large Language Models." ACL 2024.

**Important Note**: The original Chain of Logic paper focused on legal reasoning tasks and did not provide any implementation code. Our work represents the first complete implementation of the Chain of Logic methodology, developed independently from scratch based on the paper description, and extended to the domain of conditional and modal logical reasoning.

### Contact

For questions regarding this research or the experimental data, please contact: fatemehshahrokhshahi@stu.aydin.edu.tr

### License

MIT License - see LICENSE file for details.

---

### Troubleshooting

**Common Issues:**

1. **Missing prompt files**: Ensure JSON prompt files are in the correct directory structure
2. **API rate limits**: The response generator includes rate limiting (1 second delays)
3. **API key errors**: Verify API keys are correctly set in `COL-response-generator.py`
4. **Path configuration errors**: Ensure all directory paths are correctly set in both `Analysis.py` and `COL-response-generator.py`
   - Update `data_dir` and `output_dir` in `Analysis.py`
   - Update `base_path` and `prompts_base_path` in `COL-response-generator.py`
5. **Missing dependencies**: Run `pip install -r requirements.txt` to install all required packages

**Data Quality Notes:**
- 405B model responses may be limited due to API availability issues during data collection
- The analysis script handles missing data gracefully and provides data completeness summaries
- Statistical tests account for unbalanced datasets across different models
