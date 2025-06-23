import os
import json
import time
from datetime import datetime
from pathlib import Path

# Try to import each API client, skip if not available
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("OpenAI not available - install with: pip install openai")

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("Anthropic not available - install with: pip install anthropic")

try:
    import google.generativeai as genai
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False
    print("Google AI not available - install with: pip install google-generativeai")

try:
    from together import Together
    TOGETHER_AVAILABLE = True
except ImportError:
    TOGETHER_AVAILABLE = False
    print("Together not available - install with: pip install together")

try:
    from mistralai.client import MistralClient
    from mistralai.models.chat_completion import ChatMessage
    MISTRAL_AVAILABLE = True
except ImportError:
    MISTRAL_AVAILABLE = False
    print("Mistral not available - install with: pip install mistralai")

# Set up API keys (replace with your actual keys)
OPENAI_API_KEY = 'your_openai_key_here'
ANTHROPIC_API_KEY = 'your_anthropic_key_here'
GOOGLE_API_KEY = 'your_google_key_here'
TOGETHER_API_KEY = 'your_together_key_here'
MISTRAL_API_KEY = 'your_mistral_key_here'

# Initialize clients only if available
if OPENAI_AVAILABLE:
    openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
if ANTHROPIC_AVAILABLE:
    anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
if GOOGLE_AVAILABLE:
    genai.configure(api_key=GOOGLE_API_KEY)
if TOGETHER_AVAILABLE:
    together_client = Together(api_key=TOGETHER_API_KEY)
if MISTRAL_AVAILABLE:
    mistral_client = MistralClient(api_key=MISTRAL_API_KEY)

# Define rule-specific system prompts
RULE_PROMPTS = {
    "MP": """In response to the following question, think step by step using this logical reasoning method and explain your reasoning, starting your response with 'Explanation: '; then *after* explaining your reasoning, when you are ready to answer, simply write 'Answer: ' followed by 'yes' or 'no' and nothing else.

Use this 6-step reasoning process:

Step 1: Structured Input - Identify the premises and question clearly
Step 2: Rule Decomposition - Break down the conditional rule into elements (A, B)
Step 3: Logical Expression - Write the logical form (A → B) ∧ A ⊢ B?
Step 4: Question Answering - Check if the antecedent A is true, and what the conditional tells us
Step 5: Element Recomposition - Combine the answers to evaluate the logic
Step 6: Resolve Expression - Apply Modus Ponens rule to reach conclusion

CRITICAL RULE: MODUS PONENS IS VALID
- "If A, then B" and "A" implies "B"
- (A → B) ∧ A ⊢ B
- This is a fundamental rule of deductive reasoning

Now apply this reasoning to the following question:""",

    "MT": """In response to the following question, think step by step using this logical reasoning method and explain your reasoning, starting your response with 'Explanation: '; then *after* explaining your reasoning, when you are ready to answer, simply write 'Answer: ' followed by 'yes' or 'no' and nothing else.

Use this 6-step reasoning process:

Step 1: Structured Input - Identify the premises and question clearly
Step 2: Rule Decomposition - Break down the conditional rule into elements (A, B)
Step 3: Logical Expression - Write the logical form (A → B) ∧ ¬B ⊢ ¬A?
Step 4: Question Answering - Check if the consequent B is false, and what the conditional tells us
Step 5: Element Recomposition - Combine the answers to evaluate the logic
Step 6: Resolve Expression - Apply Modus Tollens rule to reach conclusion

CRITICAL RULE: MODUS TOLLENS IS VALID
- "If A, then B" and "¬B" implies "¬A"
- (A → B) ∧ ¬B ⊢ ¬A
- This is a fundamental rule of deductive reasoning

Now apply this reasoning to the following question:""",

    "AC": """In response to the following question, think step by step using this logical reasoning method and explain your reasoning, starting your response with 'Explanation: '; then *after* explaining your reasoning, when you are ready to answer, simply write 'Answer: ' followed by 'yes' or 'no' and nothing else.

Use this 6-step reasoning process:

Step 1: Structured Input - Identify the conditional statement and the consequent affirmation
Step 2: Rule Decomposition - Break down (p → q) and q components  
Step 3: Logical Expression - Analyze: Does (p → q) ∧ q ⊢ p?
Step 4: Question Answering - Test if affirming the consequent is valid
Step 5: Element Recomposition - CRITICAL: Examine the directional nature of conditionals and alternative causes
Step 6: Resolve Expression - Determine if this Affirming the Consequent inference is valid

CRITICAL RULE: AFFIRMING THE CONSEQUENT IS INVALID
- "If A, then B" and "B" does NOT imply "A"
- (p → q) ∧ q does NOT entail p
- This is a fundamental logical fallacy
- Conditionals work in one direction only
- Multiple causes can lead to the same effect

Now apply this reasoning to the following question:""",

    "DA": """In response to the following question, think step by step using this logical reasoning method and explain your reasoning, starting your response with 'Explanation: '; then *after* explaining your reasoning, when you are ready to answer, simply write 'Answer: ' followed by 'yes' or 'no' and nothing else.

Use this 6-step reasoning process:

Step 1: Structured Input - Identify the premises and question clearly
Step 2: Rule Decomposition - Break down the conditional rule into elements (A, B)
Step 3: Logical Expression - Write the logical form (A → B) ∧ ¬A ⊢ ¬B?
Step 4: Question Answering - Check if knowing A is false allows us to conclude B is false
Step 5: Element Recomposition - CRITICAL: Analyze if there are other ways B could be true without A
Step 6: Resolve Expression - Determine if this is the "Denying the Antecedent" FALLACY

CRITICAL RULE: DENYING THE ANTECEDENT IS INVALID
- "If A, then B" and "¬A" does NOT imply "¬B"
- (A → B) ∧ ¬A does NOT entail ¬B
- This is a fundamental logical fallacy
- B could be true for many other reasons besides A

Now apply this reasoning to the following question:""",

    "CONV": """In response to the following question, think step by step using this logical reasoning method and explain your reasoning, starting your response with 'Explanation: '; then *after* explaining your reasoning, when you are ready to answer, simply write 'Answer: ' followed by 'yes' or 'no' and nothing else.

Use this 6-step reasoning process:

Step 1: Structured Input - Identify the original conditional statement and what we're asked to infer
Step 2: Rule Decomposition - Break down the conditional into elements (A, B) and identify the converse
Step 3: Logical Expression - Write the logical form: (A → B) → (B → A)?
Step 4: Question Answering - Check if the original conditional proves its converse
Step 5: Element Recomposition - CRITICAL: Analyze if B could be true without A being the cause
Step 6: Resolve Expression - Determine if this is the "Conversion" FALLACY

CRITICAL RULE: CONVERSION IS INVALID
- "If A, then B" does NOT imply "If B, then A"
- (A → B) does NOT entail (B → A)
- This is a fundamental logical fallacy
- The converse is not logically equivalent to the original

Now apply this reasoning to the following question:""",

    "INV": """In response to the following question, think step by step using this logical reasoning method and explain your reasoning, starting your response with 'Explanation: '; then *after* explaining your reasoning, when you are ready to answer, simply write 'Answer: ' followed by 'yes' or 'no' and nothing else.

Use this 6-step reasoning process:

Step 1: Structured Input - Identify the original conditional statement and what we're asked to infer
Step 2: Rule Decomposition - Break down the conditional into elements (A, B) and identify the inverse
Step 3: Logical Expression - Write the logical form: (A → B) → (¬A → ¬B)?
Step 4: Question Answering - Check if the original conditional proves its inverse
Step 5: Element Recomposition - CRITICAL: Analyze if B could still be true even when A is false
Step 6: Resolve Expression - Determine if this is the "Inversion" FALLACY

CRITICAL RULE: INVERSION IS INVALID
- "If A, then B" does NOT imply "If ¬A, then ¬B"
- (A → B) does NOT entail (¬A → ¬B)
- This is a fundamental logical fallacy
- The inverse is not logically equivalent to the original

Now apply this reasoning to the following question:""",

    "AS": """In response to the following question, think step by step using this logical reasoning method and explain your reasoning, starting your response with 'Explanation: '; then *after* explaining your reasoning, when you are ready to answer, simply write 'Answer: ' followed by 'yes' or 'no' and nothing else.

Use this 6-step reasoning process:

Step 1: Structured Input - Identify the original conditional statement and the proposed strengthened antecedent
Step 2: Rule Decomposition - Break down (p → q) and ((p ∧ r) → q) components  
Step 3: Logical Expression - Analyze: Does (p → q) ⊢ ((p ∧ r) → q)?
Step 4: Question Answering - Test if antecedent strengthening is valid
Step 5: Element Recomposition - CRITICAL: Examine how additional conditions can interfere with implications
Step 6: Resolve Expression - Determine if this Antecedent Strengthening is valid

CRITICAL RULE: ANTECEDENT STRENGTHENING IS INVALID
- "If A, then B" does NOT imply "If A and C, then B"
- (p → q) does NOT entail ((p ∧ r) → q)
- This is a fundamental logical fallacy
- Additional conditions can interfere with the original implication
- Strengthening the antecedent weakens the inference

Now apply this reasoning to the following question:""",

    "CT": """In response to the following question, think step by step using this logical reasoning method and explain your reasoning, starting your response with 'Explanation: '; then *after* explaining your reasoning, when you are ready to answer, simply write 'Answer: ' followed by 'yes' or 'no' and nothing else.

Use this 6-step reasoning process:

Step 1: Structured Input - Identify the original conditional statement and what we're asked to infer
Step 2: Rule Decomposition - Break down the conditional into elements (A, B) and identify the contrapositive
Step 3: Logical Expression - Write the logical form: (A → B) ≡ (¬B → ¬A)?
Step 4: Question Answering - Apply the fundamental rule of contraposition
Step 5: Element Recomposition - Verify the logical transformation is correct
Step 6: Resolve Expression - Conclude based on the universal validity of contraposition

CRITICAL RULE: CONTRAPOSITION IS ALWAYS VALID
- "If A, then B" ALWAYS implies "If ¬B, then ¬A"
- (A → B) ≡ (¬B → ¬A)
- This is a fundamental law of logic, equivalent to the original statement

Now apply this reasoning to the following question:""",

    "CMP": """In response to the following question, think step by step using this logical reasoning method and explain your reasoning, starting your response with 'Explanation: '; then *after* explaining your reasoning, when you are ready to answer, simply write 'Answer: ' followed by 'yes' or 'no' and nothing else.

Use this 6-step reasoning process:

Step 1: Structured Input - Identify the nested conditional structure and probability claims
Step 2: Rule Decomposition - Break down into elements: probabilities, nested conditionals, and the question
Step 3: Logical Expression - Analyze: If p → (q → r) and p is likely, does q → r become likely?
Step 4: Question Answering - Check if likelihood transfers through nested conditionals
Step 5: Element Recomposition - CRITICAL: Analyze probability distribution in the conditional scenario
Step 6: Resolve Expression - Determine if this Complex Modus Ponens inference is valid

CRITICAL RULE: COMPLEX MODUS PONENS WITH PROBABILITIES IS OFTEN INVALID
- While p → (q → r), p ⊢ q → r works in pure logic, it FAILS with probability/likelihood
- When p is likely but q → r involves low-probability outcomes, the inference breaks down
- The nested conditional creates a probability distribution problem
- This is McGee's counterexample to modus ponens in probabilistic contexts

Now apply this reasoning to the following question:""",

    "DSmu": """In response to the following question, think step by step using this logical reasoning method and explain your reasoning, starting your response with 'Explanation: '; then *after* explaining your reasoning, when you are ready to answer, simply write 'Answer: ' followed by 'yes' or 'no' and nothing else.

Use this 6-step reasoning process:

Step 1: Structured Input - Identify the disjunction and modal statements
Step 2: Rule Decomposition - Break down disjunction (A ∨ □B) and modal necessity (¬□B)
Step 3: Logical Expression - Analyze: Does (A ∨ □B) ∧ ¬□B ⊢ A?
Step 4: Question Answering - Test if negated modal necessity supports disjunctive syllogism
Step 5: Element Recomposition - CRITICAL: Examine the modal scope and logical relationships
Step 6: Resolve Expression - Determine if this Disjunctive Syllogism with 'must' is valid

CRITICAL RULE: DISJUNCTIVE SYLLOGISM WITH 'MUST' IS INVALID
- Classic Disjunctive Syllogism: (A ∨ B) ∧ ¬B ⊢ A is VALID
- But with modals: (A ∨ □B) ∧ ¬□B ⊢ A is INVALID
- "Not necessarily B" does not imply "not B"
- Modal uncertainty allows both options to remain open

Now apply this reasoning to the following question:""",

    "DSmi": """In response to the following question, think step by step using this logical reasoning method and explain your reasoning, starting your response with 'Explanation: '; then *after* explaining your reasoning, when you are ready to answer, simply write 'Answer: ' followed by 'yes' or 'no' and nothing else.

Use this 6-step reasoning process:

Step 1: Structured Input - Identify the disjunction and modal statements
Step 2: Rule Decomposition - Break down disjunction (A ∨ □B) and modal possibility (♢¬B)
Step 3: Logical Expression - Analyze: Does (A ∨ □B) ∧ ♢¬B ⊢ A?
Step 4: Question Answering - Test if modal possibility eliminates disjunctive options
Step 5: Element Recomposition - CRITICAL: Distinguish "might not" from "not"
Step 6: Resolve Expression - Determine if this Disjunctive Syllogism with modals is valid

CRITICAL RULE: "MIGHT NOT" DOES NOT ELIMINATE POSSIBILITIES
- Classic Disjunctive Syllogism: (A ∨ B) ∧ ¬B ⊢ A is VALID
- But with modals: (A ∨ □B) ∧ ♢¬B ⊢ A is INVALID
- "Might not be in the garden" ≠ "Is not in the garden"
- Modal possibility allows for uncertainty - both options remain open
- This is a common modal reasoning fallacy

Now apply this reasoning to the following question:""",

    "MTmu": """In response to the following question, think step by step using this logical reasoning method and explain your reasoning, starting your response with 'Explanation: '; then *after* explaining your reasoning, when you are ready to answer, simply write 'Answer: ' followed by 'yes' or 'no' and nothing else.

Use this 6-step reasoning process:

Step 1: Structured Input - Identify the conditional with modal "must" and the negation of necessity
Step 2: Rule Decomposition - Break down p → □q and ¬□q components
Step 3: Logical Expression - Analyze: Does (p → □q) ∧ ¬□q ⊢ ¬p?
Step 4: Question Answering - Test if negated modal necessity supports modus tollens
Step 5: Element Recomposition - CRITICAL: Examine the modal scope and logical relationships
Step 6: Resolve Expression - Determine if this Modal Modus Tollens is valid

CRITICAL RULE: MODAL MODUS TOLLENS WITH 'MUST' IS VALID
- Classic Modus Tollens: (p → q) ∧ ¬q ⊢ ¬p is VALID
- Modal version: (p → □q) ∧ ¬□q ⊢ ¬p is VALID
- "Not necessarily q" contradicts "if p then necessarily q"

Now apply this reasoning to the following question:""",

    "MTmi": """In response to the following question, think step by step using this logical reasoning method and explain your reasoning, starting your response with 'Explanation: '; then *after* explaining your reasoning, when you are ready to answer, simply write 'Answer: ' followed by 'yes' or 'no' and nothing else.

Use this 6-step reasoning process:

Step 1: Structured Input - Identify the conditional with modal "must" and the modal possibility "might not"
Step 2: Rule Decomposition - Break down p → □q and ♢¬q components
Step 3: Logical Expression - Analyze: Does (p → □q) ∧ ♢¬q ⊢ ¬p?
Step 4: Question Answering - Test if modal possibility contradicts modal necessity
Step 5: Element Recomposition - CRITICAL: Distinguish "might not" from "not" in modal contexts
Step 6: Resolve Expression - Determine if this Modal Modus Tollens with 'might' is valid

CRITICAL RULE: MODAL MODUS TOLLENS WITH 'MIGHT' IS INVALID
- Classic Modus Tollens: (p → q) ∧ ¬q ⊢ ¬p is VALID
- Modal version with "not must": (p → □q) ∧ ¬□q ⊢ ¬p is VALID
- But Modal version with "might not": (p → □q) ∧ ♢¬q ⊢ ¬p is INVALID
- "Might not" expresses possibility, not definitive negation

Now apply this reasoning to the following question:""",

    "MuDistOr": """In response to the following question, think step by step using this logical reasoning method and explain your reasoning, starting your response with 'Explanation: '; then *after* explaining your reasoning, when you are ready to answer, simply write 'Answer: ' followed by 'yes' or 'no' and nothing else.

Use this 6-step reasoning process:

Step 1: Structured Input - Identify the original modal statement and the proposed distribution
Step 2: Rule Decomposition - Break down □(p ∨ q) and □p ∨ □q components
Step 3: Logical Expression - Analyze: Does □(p ∨ q) ⊢ □p ∨ □q?
Step 4: Question Answering - Test if modal necessity distributes over disjunction
Step 5: Element Recomposition - CRITICAL: Examine the scope of modal operators and logical fallacy
Step 6: Resolve Expression - Determine if this Modal Distribution is valid

CRITICAL RULE: NECESSITY DOES NOT DISTRIBUTE OVER DISJUNCTION
- "Must (A or B)" ≠ "Must A or Must B"
- □(p ∨ q) does NOT imply □p ∨ □q
- This is a fundamental modal logic fallacy
- The scope of the necessity operator matters crucially

Now apply this reasoning to the following question:""",

    "MiAg": """In response to the following question, think step by step using this logical reasoning method and explain your reasoning, starting your response with 'Explanation: '; then *after* explaining your reasoning, when you are ready to answer, simply write 'Answer: ' followed by 'yes' or 'no' and nothing else.

Use this 6-step reasoning process:

Step 1: Structured Input - Identify the separate modal possibilities and the proposed agglomeration
Step 2: Rule Decomposition - Break down ♢p ∧ ♢q and ♢(p ∧ q) components
Step 3: Logical Expression - Analyze: Does ♢p ∧ ♢q ⊢ ♢(p ∧ q)?
Step 4: Question Answering - Test if separate possibilities can be combined
Step 5: Element Recomposition - CRITICAL: Examine possibility spaces and modal independence
Step 6: Resolve Expression - Determine if this Modal Agglomeration is valid

CRITICAL RULE: POSSIBILITY DOES NOT AGGLOMERATE OVER CONJUNCTION
- "Might A and might B" ≠ "Might (A and B)"
- ♢p ∧ ♢q does NOT imply ♢(p ∧ q)
- This is the modal agglomeration fallacy
- Separate possibilities don't guarantee joint possibility

Now apply this reasoning to the following question:""",

    "NSFC": """In response to the following question, think step by step using this logical reasoning method and explain your reasoning, starting your response with 'Explanation: '; then *after* explaining your reasoning, when you are ready to answer, simply write 'Answer: ' followed by 'yes' or 'no' and nothing else.

Use this 6-step reasoning process:

Step 1: Structured Input - Identify the original modal statement and the proposed inference
Step 2: Rule Decomposition - Break down ♢(p ∨ q) and ♢p ∧ ♢q components
Step 3: Logical Expression - Analyze: Does ♢(p ∨ q) ⊢ ♢p ∧ ♢q?
Step 4: Question Answering - Test if narrow-scope free choice is valid
Step 5: Element Recomposition - CRITICAL: Examine the scope of modal operators and free choice reasoning
Step 6: Resolve Expression - Determine if this Narrow-Scope Free Choice inference is valid

CRITICAL RULE: NARROW-SCOPE FREE CHOICE IS VALID IN NATURAL LANGUAGE
- "Might (A or B)" often implies "Might A and might B" in natural language
- ♢(p ∨ q) typically supports ♢p ∧ ♢q in practical reasoning
- This is called "Free Choice" semantics - both options are genuinely available

Now apply this reasoning to the following question:"""
}

def get_available_models():
    """Get model configurations with proper naming convention - only working models"""
    available_models = {}
    
    # Only use Together API models that are working
    if TOGETHER_AVAILABLE:
        available_models.update({
            'meta_llama_Meta_Llama_3_1_8B_Instruct_Turbo': {'provider': 'together', 'model': 'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo'},
            'meta_llama_Meta_Llama_3_1_70B_Instruct_Turbo': {'provider': 'together', 'model': 'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo'},
            'meta_llama_Meta_Llama_3_1_405B_Instruct_Turbo': {'provider': 'together', 'model': 'meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo'},
        })
    
    # Add other working models here as you test them
    # Commented out problematic APIs:
    
    # if OPENAI_AVAILABLE:
    #     available_models.update({
    #         'gpt-3.5-turbo': {'provider': 'openai', 'model': 'gpt-3.5-turbo'},
    #         'gpt-4o-mini': {'provider': 'openai', 'model': 'gpt-4o-mini'},
    #         'gpt-4o': {'provider': 'openai', 'model': 'gpt-4o'},
    #     })
    
    # if ANTHROPIC_AVAILABLE:
    #     available_models.update({
    #         'claude_3_5_sonnet_20240620': {'provider': 'anthropic', 'model': 'claude-3-5-sonnet-20240620'},
    #         'claude_3_haiku_20240307': {'provider': 'anthropic', 'model': 'claude-3-haiku-20240307'},
    #     })
    
    # if GOOGLE_AVAILABLE:
    #     available_models.update({
    #         'gemini_1_5_flash': {'provider': 'google', 'model': 'gemini-1.5-flash'},
    #         'gemini_1_5_pro': {'provider': 'google', 'model': 'gemini-1.5-pro'},
    #     })
    
    # if MISTRAL_AVAILABLE:
    #     available_models.update({
    #         'mistral_small_latest': {'provider': 'mistral', 'model': 'mistral-small-latest'},
    #         'mistral_large_latest': {'provider': 'mistral', 'model': 'mistral-large-latest'},
    #     })
    
    return available_models

def load_prompts_from_json(prompts_file_path):
    """Load prompts from JSON file in the format [["Rule", number, "prompt"], ...]"""
    try:
        with open(prompts_file_path, 'r', encoding='utf-8') as f:
            prompts_data = json.load(f)
        
        prompts = {}
        for item in prompts_data:
            if len(item) >= 3:
                rule_name = item[0]
                prompt_number = item[1]
                prompt_text = item[2]
                
                if rule_name not in prompts:
                    prompts[rule_name] = {}
                prompts[rule_name][prompt_number] = prompt_text
        
        return prompts
    except Exception as e:
        print(f"Error loading prompts from {prompts_file_path}: {e}")
        return {}

def call_api(provider, model_name, system_prompt, user_prompt, temperature=0):
    """Generic API call function"""
    try:
        if provider == 'openai':
            response = openai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=2000
            )
            return response.choices[0].message.content
        
        elif provider == 'anthropic':
            response = anthropic_client.messages.create(
                model=model_name,
                max_tokens=2000,
                temperature=temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            return response.content[0].text
        
        elif provider == 'google':
            model = genai.GenerativeModel(model_name)
            prompt = f"{system_prompt}\n\n{user_prompt}"
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(temperature=temperature)
            )
            return response.text
        
        elif provider == 'together':
            response = together_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=2000
            )
            return response.choices[0].message.content
        
        elif provider == 'mistral':
            messages = [
                ChatMessage(role="system", content=system_prompt),
                ChatMessage(role="user", content=user_prompt)
            ]
            response = mistral_client.chat(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=2000
            )
            return response.choices[0].message.content
        
        else:
            return f"Unknown provider: {provider}"
    
    except Exception as e:
        return f"Error: {str(e)}"

def create_directory_structure(base_path, method_name, model_name, rule_name):
    """Create the directory structure: data/{method_name}/{model_name}/{rule_name}/"""
    path = Path(base_path) / "data" / method_name / model_name / rule_name
    path.mkdir(parents=True, exist_ok=True)
    return path

def test_single_prompt(model_config, rule_name, prompt_number, prompt_text, system_prompt):
    """Test a single prompt and return the result"""
    provider = model_config['provider']
    model_name = model_config['model']
    
    try:
        response = call_api(provider, model_name, system_prompt, prompt_text)
        
        if response.startswith("Error:"):
            return None, response
        
        # Create the result structure matching the original format
        result = {
            "user_prompt": prompt_text,
            "system_prompt": system_prompt,
            "model": model_name,
            "temperature": 0,
            "responses": [
                {
                    "content": response
                }
            ]
        }
        
        return result, None
    
    except Exception as e:
        return None, f"Exception: {str(e)}"

def main():
    """Main function to test all models and rules"""
    base_path = r"C:\Users\Lenovo\Desktop\Rule ML+RR\conditional and modal reasoning in LLMs\Test"
    method_name = "chain_of_logic"  # or "chain_of_thought"
    
    # Load all prompts from files
    rules_to_test = ["MP", "MT", "AC", "DA", "CONV", "INV", "AS", "CT", "CMP", "DSmu", "DSmi", "MTmu", "MTmi", "MuDistOr", "MiAg", "NSFC"]
    prompts_base_path = r"C:\Users\Lenovo\Desktop\Rule ML+RR\conditional and modal reasoning in LLMs\llm-logic-main\llm-logic-main\prompts"
    
    all_prompts = {}
    for rule_name in rules_to_test:
        prompts_file = os.path.join(prompts_base_path, f"{rule_name}.json")
        print(f"Looking for prompts file: {prompts_file}")
        if os.path.exists(prompts_file):
            print(f"Found prompts file for {rule_name}")
            rule_prompts = load_prompts_from_json(prompts_file)
            if rule_prompts:
                all_prompts.update(rule_prompts)
                print(f"Loaded {len(rule_prompts)} rule sets from {rule_name}")
            else:
                print(f"No prompts loaded from {rule_name}")
        else:
            print(f"Warning: Prompts file not found for {rule_name}: {prompts_file}")
            # Check if files exist in current directory as fallback
            local_file = f"{rule_name}.json"
            if os.path.exists(local_file):
                print(f"Found {rule_name}.json in current directory, using that instead")
                rule_prompts = load_prompts_from_json(local_file)
                if rule_prompts:
                    all_prompts.update(rule_prompts)
                    print(f"Loaded {len(rule_prompts)} rule sets from local {rule_name}.json")
    
    if not all_prompts:
        print("No prompts loaded. Exiting.")
        return
    
    MODEL_CONFIGS = get_available_models()
    if not MODEL_CONFIGS:
        print("No models available! Please install the required packages.")
        return
    
    print(f"Starting Chain of Logic testing...")
    print(f"Available APIs: OpenAI={OPENAI_AVAILABLE}, Anthropic={ANTHROPIC_AVAILABLE}, Google={GOOGLE_AVAILABLE}, Together={TOGETHER_AVAILABLE}, Mistral={MISTRAL_AVAILABLE}")
    print(f"Testing {len(MODEL_CONFIGS)} models on {len(all_prompts)} rules...")
    
    total_tests = 0
    successful_tests = 0
    
    # Test each model on each rule
    for model_key, model_config in MODEL_CONFIGS.items():
        print(f"\n--- Testing model: {model_key} ---")
        
        for rule_name, rule_prompts in all_prompts.items():
            if rule_name not in RULE_PROMPTS:
                print(f"Warning: No system prompt defined for rule {rule_name}, skipping...")
                continue
            
            print(f"  Testing rule: {rule_name}")
            system_prompt = RULE_PROMPTS[rule_name]
            
            # Create directory for this model and rule
            output_dir = create_directory_structure(base_path, method_name, model_key, rule_name)
            
            # Test each prompt for this rule
            for prompt_number, prompt_text in rule_prompts.items():
                total_tests += 1
                
                print(f"    Testing prompt {prompt_number}...")
                
                result, error = test_single_prompt(
                    model_config, rule_name, prompt_number, prompt_text, system_prompt
                )
                
                if result:
                    # Save individual result file
                    output_file = output_dir / f"{rule_name}_{prompt_number}_0.json"
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(result, f, indent=2, ensure_ascii=False)
                    
                    successful_tests += 1
                    print(f"    ✓ Saved to {output_file}")
                else:
                    print(f"    ✗ Error: {error}")
                
                # Add delay to avoid rate limits
                time.sleep(1)
    
    print(f"\n--- Testing Complete ---")
    print(f"Total tests: {total_tests}")
    print(f"Successful tests: {successful_tests}")
    print(f"Success rate: {successful_tests/total_tests*100:.1f}%")
    
    print(f"\nFiles saved in structure:")
    print(f"{base_path}/data/{method_name}/{{model_name}}/{{rule_name}}/{{rule_name}}_{{number}}_0.json")

if __name__ == "__main__":
    main()