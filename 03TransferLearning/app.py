import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import pandas as pd

# Number of classes in your model
NUM_CLASSES = 15

# Optionally define class names (update as needed)
class_names =["bedroom",
              "coast",
              "forest",
              "highway",
              "industrial",
              "Inside City",
              "kitchen",
              "living room",
              "mountain",
              "office",
              "Open country",
              "store","Street",
              "Suburb",
              "Tall building"]


def load_model(model_path):
    """Load the EfficientNet B3 model with a custom classifier."""
    model = models.efficientnet_b3(pretrained=False)
    model.classifier = nn.Sequential(
        nn.Dropout(0.5, inplace=True),
        nn.Linear(1536, NUM_CLASSES, bias=True),
    )
    # Load weights
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model

def preprocess_image(image: Image.Image):
    """Preprocess image to match model input dimensions."""
    preprocess = transforms.Compose([
        transforms.Grayscale(3),
        transforms.Resize(300),
        transforms.CenterCrop(300),
        transforms.ToTensor(),
        # Uncomment and adjust if your model was trained with normalization:
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(image)

def main():
    # Set the page configuration
    st.set_page_config(page_title="Image Classification", layout="centered")

    # Title and instructions
    st.title("Image Classification with Pretrained Model")
    st.write(
        "Follow the steps below to load your model, upload an image, "
        "and view the top 5 predictions."
    )

    # Step 1: Enter the path to your pretrained model
    st.subheader("1. Load Pretrained Model")
    model_path = st.text_input(
        label="Enter the path to your .pth model file:",
        value="model.pth"
    )
    
    if st.button("Load Model"):
        try:
            model = load_model(model_path)
            st.success("Model loaded successfully!")
            st.session_state["model"] = model
        except Exception as e:
            st.error(f"Error loading model: {e}")

    # Proceed only if a model is loaded
    if "model" in st.session_state:
        st.divider()  # Just a visual separator (Streamlit >= 1.20)
        
        # Step 2: Upload an image
        st.subheader("2. Upload an Image")
        uploaded_file = st.file_uploader(
            "Choose an image (jpg, jpeg, png):",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)
        else:
            st.info("Please upload an image to proceed.")

        st.divider()

        # Step 3: Run inference
        st.subheader("3. Run Inference")
        if st.button("Run Inference"):
            if uploaded_file is None:
                st.warning("You must upload an image before running inference.")
            else:
                # Preprocessing
                with st.spinner("Preprocessing image..."):
                    input_tensor = preprocess_image(image)
                    input_batch = input_tensor.unsqueeze(0)

                # Inference
                with st.spinner("Running inference..."):
                    with torch.no_grad():
                        output = st.session_state["model"](input_batch)
                    probabilities = F.softmax(output, dim=1)

                    # Get top 5 predictions
                    top5_prob, top5_indices = torch.topk(probabilities, k=5, dim=1)
                    top5_prob = top5_prob.squeeze().tolist()
                    top5_indices = top5_indices.squeeze().tolist()

                # Display results
                st.markdown("### Top 5 Predictions")
                results = []
                for prob, idx in zip(top5_prob, top5_indices):
                    class_name = class_names[idx] if idx < len(class_names) else f"Class {idx}"
                    results.append((class_name, f"{prob * 100:.2f}%"))
                
                df = pd.DataFrame(results, columns=["Class", "Probability"])
                st.table(df)

if __name__ == "__main__":
    main()
