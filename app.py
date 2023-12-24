from transformers import pipeline
import gradio as gr

#load the model directly
# Use a pipeline as a high-level helper
pipe = pipeline("text-classification", model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")


#run the application
demo=gr.Interface.from_pipeline(pipe)
demo.launch()



# import gradio as gr
# from transformers import AutoModelForSequenceClassification, AutoTokenizer
# import torch

# # Load the pre-trained model and tokenizer
# tokenizer = AutoTokenizer.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
# model = AutoModelForSequenceClassification.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")

# # Define a function for sentiment analysis
# def predict_sentiment(text):
#     # Tokenize the input text and prepare it to be used by the model
#     inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

#     # Forward pass through the model
#     with torch.no_grad():
#         outputs = model(**inputs)

#     # Get the predicted probabilities and convert them to percentages
#     probabilities = torch.softmax(outputs.logits, dim=1).squeeze().tolist()
#     positive_percent = probabilities[2] * 100
#     negative_percent = probabilities[0] * 100
#     neutral_percent = probabilities[1] * 100

#     # Construct the result dictionary
#     result = {
#         "Positive": round(positive_percent, 2),
#         "Negative": round(negative_percent, 2),
#         "Neutral": round(neutral_percent, 2)
#     }

#     return result

# # Define inputs and outputs directly without using gr.inputs or gr.outputs
# iface = gr.Interface(
#     fn=predict_sentiment,
#     inputs=gr.inputs.Textbox(lines=10, label="Enter financial statement"),
#     outputs=gr.outputs.Label(num_top_classes=3, label="Sentiment Percentages"),
#     title="Financial Statement Sentiment Analysis",
#     description="Predict the sentiment percentages of a financial statement."
# )

# if __name__ == "__main__":
#     iface.launch()
