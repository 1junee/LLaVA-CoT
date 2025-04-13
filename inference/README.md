# Inference 

You have two options to run inference with LLaVA-CoT. 

1. **Run the inference script.** This is the simplest way to have a quick test.
    - In order to run the demo, you need to create a new environment with the following command:
      ```bash
      cd demo
      conda create -n llava-cot python=3.10
      conda activate llava-cot
      pip install -r requirements.txt
      ```
    - You can use the following command to run the demo:
      ```bash
      python simple_inference.py \
        --model_name_or_path "Xkev/Llama-3.2V-11B-cot" \
        --prompt "How to make this pastry?" \
        --image_path "pastry.png" \
        --type "stage"
      ```
      You are recommended to take a look at the [simple_inference.py](demo/simple_inference.py) file to see more available arguments.
    - Additionally, you need to replace the `processing_mllama.py` file in the transformers library (`YOUR_ENV/lib/python3.10/site-packages/transformers/models/mllama/processing_mllama.py`) with the one provided in [processing_mllama.py](processing_mllama.py).

2. **Run the inference using VLMEvalKit.** This supports any datasets in the VLMEvalKit. 
    - In order to run the demo, you need to replace code provided in [VLMEvalKit/inference_demo.py](VLMEvalKit/inference_demo.py) with the original inference code for Llama-3.2-11B-Vision-Instruct in VLMEvalKit.
    - Additionally, you need to replace the `processing_mllama.py` file in the transformers library (`YOUR_ENV/lib/python3.10/site-packages/transformers/models/mllama/processing_mllama.py`) with the one provided in [processing_mllama.py](processing_mllama.py).
    
