import torch
from PIL import Image
import re
import numpy as np
import copy
import argparse
from transformers import StoppingCriteria, StoppingCriteriaList, MllamaForConditionalGeneration, AutoProcessor

parser = argparse.ArgumentParser(description="LLaVA-CoT Simple Inference")
parser.add_argument(
    "--model_name_or_path",
    type=str,
    default="Xkev/Llama-3.2V-11B-cot",
    help="Path to the model.",
)
parser.add_argument(
    "--prompt",
    type=str,
    help="Prompt to ask the model.",
)
parser.add_argument(
    "--image_path",
    type=str,
    help="Path to the image.",
)
parser.add_argument(
    "--type",
    type=str,
    default="stage",
    choices=["best_of_N", "sentence", "stage"],
    help="Type of generation to perform.",
)
parser.add_argument(
    "--beam_size",
    type=int,
    default=2,
    help="Number of candidates to generate.",
)
parser.add_argument(
    "--device",
    type=str,
    default="cpu",
    help="Device to use for inference.",
)
parser.add_argument(
    "--load_in_8bit",
    action="store_true",
    help="Load the model using 8-bit quantization to reduce memory usage.",
)
args = parser.parse_args()

class StopOnStrings(StoppingCriteria):
    def __init__(self, stop_strings, tokenizer):
        self.stop_strings = stop_strings
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        for stop_string in self.stop_strings:
            if stop_string in generated_text:
                return True
        return False
    
class StopOnPeriod(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        if generated_text.endswith('.'):
            return True
        return False

model_name_or_path = args.model_name_or_path
load_kwargs = {
    "torch_dtype": torch.bfloat16,
}

if args.device == "cpu":
    load_kwargs["device_map"] = "cpu"
else:
    load_kwargs["device_map"] = "auto"

if getattr(args, "load_in_8bit", False):
    load_kwargs["load_in_8bit"] = True

try:
    model = MllamaForConditionalGeneration.from_pretrained(
        model_name_or_path,
        **load_kwargs,
    )
    model.to(args.device)
except RuntimeError as e:
    print(f"RuntimeError while loading model on {args.device}: {e}")
    print("Falling back to CPU.")
    args.device = "cpu"
    model = MllamaForConditionalGeneration.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    model.to("cpu")

model.eval()
device = args.device
processor = AutoProcessor.from_pretrained(model_name_or_path)
kwargs = dict(do_sample=True, max_new_tokens=2048, temperature=0.6, top_p=0.9)

def judge(image, prompt, outputs, type="summary"):
    input_outputs = []
    
    hint = None
    if type == "all":
        judge_prompt = f'Now you act as a judge, helping me determine which of the two texts I provide better answers the question.'
        recall_prompt = ""
        for output in outputs:
            input_outputs.append(output)
    elif type == "sentence":
        judge_prompt = f'Now you act as a judge, helping me determine which of the two texts I provide is a better next sentence for the answer to the question.'
        recall_prompt = ""
        for output in outputs:
            sentences = output.split(".")
            if len(sentences) > 2:
                hint = ' '.join(sentences[:-2])
            input_outputs.append(sentences[-2])
    elif type == "summary":
        judge_prompt = f'Now you act as a judge, helping me determine which of the two texts I provide better provides a summary of what it should do to solve the question. The summary should focus on outlining the main approach instead of stating specific analytical reasoning or math formula.'
        recall_prompt = f'Please note that a better summary should focus on outlining the main approach instead of stating specific analytical reasoning or math formula.'
        for output in outputs:
            input_match = re.search(r'<SUMMARY>(.*?)</SUMMARY>', output, re.DOTALL)
            if input_match:
                input_outputs.append(input_match.group(1))
    elif type == "caption":
        judge_prompt = f'Now you act as a judge, helping me determine which of the two texts I provide better summarizes the information in the image related to the question, and has fewer errors. It is essential that the captions are as thorough as possible while remaining accurate, capturing as many details as possible rather than providing only general commentary.'
        recall_prompt = f'Please note that a better caption should be as thorough as possible while remaining accurate, capturing as many details as possible rather than providing only general commentary.'
        for output in outputs:
            input_match = re.search(r'<CAPTION>(.*?)</CAPTION>', output, re.DOTALL)
            if input_match:
                hint_match = re.search(r'<SUMMARY>(.*?)</SUMMARY>', output, re.DOTALL)
                if hint_match:
                    input_outputs.append(input_match.group(1))
    elif type == "reasoning":
        judge_prompt = f'Now you act as a judge, helping me determine which of the two texts I provide better explains the reasoning process to solve the question, and has fewer errors. Begin by thoroughly reviewing the question, followed by an in-depth examination of each answer individually, noting any differences. Subsequently, analyze these differences to determine which response demonstrates stronger reasoning and provide a clear conclusion.'
        recall_prompt = f'Begin by thoroughly reviewing the question, followed by an in-depth examination of each answer individually, noting any differences. Subsequently, analyze these differences to determine which response demonstrates stronger reasoning and provide a clear conclusion.'
        for output in outputs:
            input_match = re.search(r'<REASONING>(.*?)</REASONING>', output, re.DOTALL)
            if input_match:
                hint_match = re.search(r'<SUMMARY>(.*?)</SUMMARY>', output, re.DOTALL)
                if hint_match:
                    hint_caption_match = re.search(r'<CAPTION>(.*?)</CAPTION>', output, re.DOTALL)
                    if hint_caption_match:
                        hint = hint_caption_match.group(1)
                        input_outputs.append(input_match.group(1))
    elif type == "conclusion":
        judge_prompt = f'Now you act as a judge, helping me determine which of the two texts I provide offers a more effective conclusion to the question. The conclusion should align with the reasoning presented in the hint. The conclusion should never refuse to answer the question.'
        recall_prompt = f'Please note that a better conclusion should align with the reasoning presented in the hint. The conclusion should never refuse to answer the question.'
        for output in outputs:
            input_match = re.search(r'<CONCLUSION>(.*?)</CONCLUSION>', output, re.DOTALL)
            if input_match:
                hint_match = re.search(r'<SUMMARY>(.*?)</SUMMARY>', output, re.DOTALL)
                if hint_match:
                    hint_caption_match = re.search(r'<CAPTION>(.*?)</CAPTION>', output, re.DOTALL)
                    if hint_caption_match:
                        hint_reasoning_match = re.search(r'<REASONING>(.*?)</REASONING>', output, re.DOTALL)
                        if hint_reasoning_match:
                            hint = hint_caption_match.group(1) + hint_reasoning_match.group(1)
                            input_outputs.append(input_match.group(1))

    if type == "reasoning":
        reasoning_prompt = f"""Now you act as a judge, helping me determine whether the reasoning process in the given text is correct and accurate based on the given information.
        You should assume that the given information about the image is correct.
        You should only consider the reasoning process itself, not the correctness of the background information.  
        If the reasoning process invovles any calculations, you should verify the accuracy of the calculations.
        You should output 'correct' if you don't find any errors in the reasoning process, and 'incorrect' if you find any errors."""
        
        reasoning_prompt_1 = reasoning_prompt + f'\n\nGiven Information: {hint}' + f'\n\nReasoning Process: {input_outputs[0]}'
        reasoning_message_1 = [
            {'role': 'user', 'content': [
                {'type': 'text', 'text': reasoning_prompt_1}
            ]}
        ]
        reasoning_input_text_1 = processor.apply_chat_template(reasoning_message_1, add_generation_prompt=True)
        reasoning_inputs_1 = processor(None, reasoning_input_text_1, return_tensors='pt').to(device)
        reasoning_output_1 = model.generate(**reasoning_inputs_1, **kwargs)
        reasoning_output_text_1 = processor.decode(reasoning_output_1[0][reasoning_inputs_1['input_ids'].shape[1]:]).replace('<|eot_id|>', '').replace('<|endoftext|>', '')
        if "incorrect" in reasoning_output_text_1:
            return 1
        
        reasoning_prompt_2 = reasoning_prompt + f'\n\nGiven Information: {hint}' + f'\n\nReasoning Process: {input_outputs[1]}'
        reasoning_message_2 = [
            {'role': 'user', 'content': [
                {'type': 'text', 'text': reasoning_prompt_2}
            ]}
        ]
        reasoning_input_text_2 = processor.apply_chat_template(reasoning_message_2, add_generation_prompt=True)
        reasoning_inputs_2 = processor(None, reasoning_input_text_2, return_tensors='pt').to(device)
        reasoning_output_2 = model.generate(**reasoning_inputs_2, **kwargs)
        reasoning_output_text_2 = processor.decode(reasoning_output_2[0][reasoning_inputs_2['input_ids'].shape[1]:]).replace('<|eot_id|>', '').replace('<|endoftext|>', '')
        if "incorrect" in reasoning_output_text_2:
            return 0
            
    judge_prompt += f'\n\nQuestion: {prompt}'
    if hint:
        judge_prompt += f'\n\nHint about the Question: {hint}'
    for i, output in enumerate(input_outputs):
        judge_prompt += f'\nRepsonse {i+1}: {output}'
    judge_prompt += f'\n\n{recall_prompt}'
    judge_prompt += f' Please strictly follow the following format requirements when outputting, and don’t have any other unnecessary words.'
    judge_prompt += f'\n\nOutput format: "Since [reason], I choose response [1/2]."'
    
    judge_message = [
        {'role': 'user', 'content': [
            {'type': 'image'},
            {'type': 'text', 'text': judge_prompt}
        ]}
    ]
    judge_input_text = processor.apply_chat_template(judge_message, add_generation_prompt=True)
    judge_inputs = processor(image, judge_input_text, return_tensors='pt').to(device)
    judge_output = model.generate(**judge_inputs, **kwargs)
    judge_output_text = processor.decode(judge_output[0][judge_inputs['input_ids'].shape[1]:]).replace('<|eot_id|>', '').replace('<|endoftext|>', '')
    
    if "I choose response 1" in judge_output_text:
        return 0
    else:
        return 1
        
def generate_inner_best_of_N(prompt, image_path, beam_size=2):

    image = Image.open(image_path)
    messages = [
        {'role': 'user', 'content': [
            {'type': 'image'},
            {'type': 'text', 'text': prompt}
        ]}
    ]
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(image, input_text, return_tensors='pt').to(device)
    
    initial_length = len(inputs['input_ids'][0])
    input_ids = copy.deepcopy(inputs['input_ids'])

    stop_criteria = StoppingCriteriaList([StopOnStrings(['</CONCLUSION>'], processor.tokenizer)])   
    candidates = []
    for _ in range(beam_size): 
        generation_kwargs = kwargs.copy()
        generation_kwargs.update({
            'stopping_criteria': stop_criteria
        })
        
        inputs = processor(image, input_ids, return_tensors='pt').to(device)
        output = model.generate(**inputs, **generation_kwargs)
        
        new_generated_ids = output[0]
        
        generated_text = processor.tokenizer.decode(new_generated_ids[initial_length:], skip_special_tokens=True)
        
        candidates.append({
            'input_ids': new_generated_ids.unsqueeze(0),
            'generated_text': generated_text,
        })
    
    while(len(candidates) > 1):
        # randomly select two candidates
        candidate1 = candidates.pop(np.random.randint(len(candidates)))
        candidate2 = candidates.pop(np.random.randint(len(candidates)))
        outputs = [candidate1['generated_text'], candidate2['generated_text']]
        best_index = judge(image, prompt, outputs, type="all")
        if best_index == 0:
            candidates.append(candidate1)
        else:
            candidates.append(candidate2)
    
    input_ids = candidates[0]['input_ids']

    final_output = processor.tokenizer.decode(input_ids[0][initial_length:], skip_special_tokens=True)
    return final_output

def generate_inner_sentence_beam(prompt, image_path, beam_size=2):
    
    image = Image.open(image_path)
    messages = [
        {'role': 'user', 'content': [
            {'type': 'image'},
            {'type': 'text', 'text': prompt}
        ]}
    ]
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(image, input_text, return_tensors='pt').to(device)

    initial_length = len(inputs['input_ids'][0])
    input_ids = copy.deepcopy(inputs['input_ids'])

    while "</CONCLUSION>" not in processor.tokenizer.decode(input_ids[0][initial_length:], skip_special_tokens=True):
        stop_criteria = StoppingCriteriaList([StopOnPeriod(processor.tokenizer), StopOnStrings(["</CONCLUSION>"], processor.tokenizer)])
        
        candidates = []
        for _ in range(beam_size):
            generation_kwargs = kwargs.copy()
            generation_kwargs.update({
                'stopping_criteria': stop_criteria
            })
            
            inputs = processor(image, input_ids, return_tensors='pt').to(device)
            output = model.generate(**inputs, **generation_kwargs)
            
            new_generated_ids = output[0]
            
            generated_text = processor.tokenizer.decode(new_generated_ids[initial_length:], skip_special_tokens=True)
            
            candidates.append({
                'input_ids': new_generated_ids.unsqueeze(0),
                'generated_text': generated_text,
            })
        
        while(len(candidates) > 1):
            # randomly select two candidates
            candidate1 = candidates.pop(np.random.randint(len(candidates)))
            candidate2 = candidates.pop(np.random.randint(len(candidates)))
            outputs = [candidate1['generated_text'], candidate2['generated_text']]
            best_index = judge(image, prompt, outputs, type="sentence")
            if best_index == 0:
                candidates.append(candidate1)
            else:
                candidates.append(candidate2)
        
        input_ids = candidates[0]['input_ids']

    final_output = processor.tokenizer.decode(input_ids[0][initial_length:], skip_special_tokens=True)
    return final_output

def generate_inner_stage_beam(prompt, image_path, beam_size=2):

    image = Image.open(image_path)
    messages = [
        {'role': 'user', 'content': [
            {'type': 'image'},
            {'type': 'text', 'text': prompt}
        ]}
    ]
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(image, input_text, return_tensors='pt').to(device)
    
    stages = ['<SUMMARY>', '<CAPTION>', '<REASONING>', '<CONCLUSION>']
    end_markers = ['</SUMMARY>', '</CAPTION>', '</REASONING>', '</CONCLUSION>']

    initial_length = len(inputs['input_ids'][0])
    input_ids = copy.deepcopy(inputs['input_ids'])

    for stage, end_marker in zip(stages, end_markers):
        stop_criteria = StoppingCriteriaList([StopOnStrings([end_marker], processor.tokenizer)])
        
        candidates = []
        for _ in range(beam_size):  
            generation_kwargs = kwargs.copy()
            generation_kwargs.update({
                'stopping_criteria': stop_criteria
            })
            
            inputs = processor(image, input_ids, return_tensors='pt').to(device)
            output = model.generate(**inputs, **generation_kwargs)
            
            new_generated_ids = output[0]
            
            generated_text = processor.tokenizer.decode(new_generated_ids[initial_length:], skip_special_tokens=True)
            
            candidates.append({
                'input_ids': new_generated_ids.unsqueeze(0),
                'generated_text': generated_text,
            })
        
        while(len(candidates) > 1):
            # randomly select two candidates
            candidate1 = candidates.pop(np.random.randint(len(candidates)))
            candidate2 = candidates.pop(np.random.randint(len(candidates)))
            outputs = [candidate1['generated_text'], candidate2['generated_text']]
            best_index = judge(image, prompt, outputs, type=stage[1:-1].lower())
            if best_index == 0:
                candidates.append(candidate1)
            else:
                candidates.append(candidate2)
        
        input_ids = candidates[0]['input_ids']

    final_output = processor.tokenizer.decode(input_ids[0][initial_length:], skip_special_tokens=True)
    return final_output

def generate_inner(prompt, image_path, type="stage", beam_size=2):
    if type == "best_of_N":
        return generate_inner_best_of_N(prompt, image_path, beam_size)
    elif type == "sentence":
        return generate_inner_sentence_beam(prompt, image_path, beam_size)
    elif type == "stage":
        return generate_inner_stage_beam(prompt, image_path, beam_size)
    else:
        raise ValueError("Invalid type. Choose from 'best_of_N', 'sentence', or 'stage'.")

print(generate_inner(args.prompt, args.image_path, type=args.type, beam_size=args.beam_size))