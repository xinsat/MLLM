from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-cascade")

pipeline.to("cpu")
pipeline("An image of a squirrel in Picasso style").images[0]