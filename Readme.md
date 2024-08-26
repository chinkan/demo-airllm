# AirLLM Demo Project

## 1. Introduction

This project demonstrates the usage of AirLLM, a powerful library for working with large language models. It showcases how to load and use a pre-trained model (Platypus2-70B-instruct) for text generation tasks.

## 2. How to Start

1. Install the required dependencies:

    ```
    pip install -r requirements.txt
    ```

2. Open and run the Jupyter notebook `main.ipynb`. The notebook contains the following key steps:

    a. Import necessary libraries and set up the environment:

```python
   from airllm import AutoModel
```

b. Load the pre-trained model:

```python
    MAX_LENGTH = 128
    # could use hugging face model repo id:\n",
    model = AutoModel.from_pretraine("garage-bAIndPlatypus2-70B-instruct")
```

c. Generate text using the loaded model:

```python
    input_text = [
        # 'What is the capital of United States?',
        'Why is the sky blue?',
        #'I like',
    ]

    input_tokens = model.tokenizer(input_text,
        return_tensors="pt",
        return_attention_mask=False,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False)

    generation_output = model.generate(
        input_tokens['input_ids'].cuda(),
        max_new_tokens=20,
        use_cache=True,
        return_dict_in_generate=True)

    output = model.tokenizer.decode(generation_output.sequences[0])

    print(output)
```

3. Experiment with different input prompts by modifying the `input_text` variable in the notebook.

## 3. Thanks

Special thanks to Gavin Li for developing AirLLM. You can find the project on GitHub at [github.com/lyogavin/airllm](https://github.com/lyogavin/airllm).
