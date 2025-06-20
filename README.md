## Supplementary Material for "Enhancing Rule-Based Reasoning in Large Language Models: A Chain of Logic Approach for Conditional and Modal Inference"

Paper by [Fatemeh Shahrokhshahi1, Farzan Mohammadi2, Hayder Mohammedqasim3,
 and Ferdi Sonmez4] ([ fatemehshahrokhshahi@stu.aydin.edu.tr
 2 farzan@blackscale.media
 3 hmohammedqasim@aydin.edu.tr
 4 ferdisonmez@aydin.edu.tr])

### Abstract

This repository contains the experimental data and analysis for evaluating the Chain of Logic prompting method on conditional and modal reasoning tasks. We extend the experimental framework originally established by Holliday et al. (2024) by implementing and evaluating the Chain of Logic prompting method introduced by Servantez et al. (2024) on the 16 logical inference patterns from the conditional and modal reasoning benchmark.

### Key Findings

Our evaluation demonstrates that Chain of Logic achieves an average accuracy of 80.84% across all models and patterns, representing substantial improvements over existing prompting methods:
- Chain of Logic: 80.84%
- Chain of Thought: 59.40% (+21.44 percentage point improvement)
- Few-Shot: 53.21% (+27.63 percentage point improvement)
- Zero-Shot: 45.71% (+35.13 percentage point improvement)

### Repository Structure
```
llm-reasoning/
├── chain-of-logic-evaluation.ipynb    # Main experiment notebook
├── prompts/                           # Logical inference pattern prompts
│   ├── AC.json                       # Affirming Consequent prompts (20 items)
│   ├── DSmu.json                     # Disjunctive Syllogism must prompts (20 items)
│   ├── MP.json                       # Modus Ponens prompts (20 items)
│   └── ...                           # 13 more rule files (16 total)
├── data/                             # Model response data
│   ├── chain_of_logic/              # Chain of Logic method responses
│   │   ├── meta_llama_Meta_Llama_3_1_8B_Instruct_Turbo/
│   │   │   ├── AC/                  # AC_1_0.json to AC_20_0.json
│   │   │   ├── DSmu/                # DSmu_1_0.json to DSmu_20_0.json
│   │   │   └── ...                  # 14 more rule directories (16 total)
│   │   ├── meta_llama_Meta_Llama_3_1_70B_Instruct_Turbo/
│   │   ├── meta_llama_Meta_Llama_3_1_405B_Instruct_Turbo/
│   │   └── ...
│   ├── chain_of_thought/            # Chain of Thought method responses
│   ├── few_shot/                    # Few-shot method responses
│   ├── zero_shot/                   # Zero-shot method responses
│   └── processed_results/           # Processed and analyzed data
├── results/                         # Experimental results and analysis
├── graphs/                          # Generated visualizations
├── src/                            # Utility functions for data processing
└── config/                         # Experimental configuration files
```

### Experimental Framework

This work builds upon the experimental framework established by Holliday et al. (2024) for evaluating conditional and modal reasoning in large language models. We extend their methodology by implementing the Chain of Logic prompting method and comparing it against their baseline approaches.

#### Models Evaluated
- Llama 3.1 Instruct 8B
- Llama 3.1 Instruct 70B  
- Llama 3.1 Instruct 405B

#### Logical Inference Patterns
16 fundamental patterns including valid inferences (Modus Ponens, Modus Tollens, Disjunctive Syllogism) and invalid patterns (Affirming the Consequent, Denying the Antecedent, Conversion), with modal variants incorporating necessity (must) and possibility (might) operators.

#### Prompting Methods Compared
- **Chain of Logic**: Six-step structured decomposition method (Servantez et al., 2024)
- **Chain of Thought**: Standard intermediate reasoning approach
- **Few-Shot**: Example-based prompting
- **Zero-Shot**: Direct prompting without examples

### Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/fatemeshahrokhshahi/llm-reasoning.git
   cd llm-reasoning
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run analysis:
   ```bash
   jupyter notebook chain-of-logic-evaluation.ipynb
   ```

### Data Organization

Response data is organized hierarchically by prompting method, model, and logical rule type:
```
data/[method]/[model]/[rule]/response_files.json
```

Example: `data/chain_of_logic/meta_llama_Meta_Llama_3_1_8B_Instruct_Turbo/AC/`

### Experimental Results

The Chain of Logic method demonstrates consistent improvements across all model sizes, with particularly notable gains for smaller models. Detailed results and statistical analysis are provided in the accompanying paper and can be reproduced using the provided notebook.

### Citation

If you use this data or build upon this work, please cite:

```bibtex
@article{llm_reasoning_2025,
  title={Enhancing Rule-Based Reasoning in Large Language Models: A Chain of Logic Approach for Conditional and Modal Inference},
  author={[Fatemeh Shahrokhshahi1, Farzan Mohammadi2, Hayder Mohammedqasim3,
 and Ferdi Sonmez4]},
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

### Contact

For questions regarding this research or the experimental data, please contact: [fatemehshahrokhshahi@stu.aydin.edu.tr]

### License

MIT License - see LICENSE file for details.
