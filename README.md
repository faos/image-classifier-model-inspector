# Model Inspector 
The Model Inspector Tool is designed to help analyze the robustness of image classification models by evaluating their performance on different versions of an image. It has three analysis modules: image processing, interpretability, and signal. The image processing module applies various noise and spatial transformations to the input image to assess the model's sensitivity to changes in visual information. The interpretability module provides visualizations of the model's decision-making process through interpretability methods and a unique U Analysis. The signal module allows users to select the image's foreground region and analyzes the model's performance on that signal compared to the background. Overall, the Model Inspector Tool is a powerful tool for evaluating the robustness of image classification models. 

# Getting Started
1. Go to the project directory and create the env with python3 -m venv env
2. Activate the env: source env/bin/activate
3. Install the project dependencies: pip install -r requirements.txt
4. streamlit run Main.py

## Deploy with docker

### Build docker image
docker build -t <image-name>:<image-tag> .
Example: docker build -t modelinspector:test .

### Run the docker image

docker run --network host --rm <image-name>:<image-tag>
Example: docker run --network host --rm modelinspector:test

