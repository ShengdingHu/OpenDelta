from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased") # suppose we load BART

from opendelta import Visualization
Visualization(model).structure_graph()