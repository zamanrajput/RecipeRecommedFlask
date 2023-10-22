import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import os

# Step 1: Load the dataset
dataset = pd.read_csv('Book9 9.csv')

# Assuming your dataset has 'recipe_id', 'ingredients'c, 'cooking_instructions', 'recipe_name', and 'image_path' columns
recipe_ids = dataset['recipe_id']
ingredients = dataset['ingredients']
cooking_instructions = dataset['cooking_instructions']
recipe_names = dataset['recipe_name']
net_price = dataset['Net Price']
image_paths = dataset['image_path']  # Assuming 'image_path' contains local file paths

# Step 2: Preprocess the ingredients using TF-IDF vectorization
vectorizer = TfidfVectorizer()
ingredient_vectors = vectorizer.fit_transform(ingredients)


# Step 3: Define a function to recommend recipes based on input ingredients
def recommend_recipes(input_ingredients, k=5):
    input_ingredients = [ingredient.strip().lower() for ingredient in
                         input_ingredients]  # Convert to lowercase and trim spaces
    input_ingredients_str = ', '.join(input_ingredients)
    input_vector = vectorizer.transform([input_ingredients_str])
    similarity_scores = cosine_similarity(input_vector, ingredient_vectors)[0]
    indices = similarity_scores.argsort()[::-1]

    recommended_recipes = []
    for idx in indices:
        recipe_ingredients = [ingredient.strip().lower() for ingredient in ingredients[idx].split(',')]
        if set(input_ingredients) <= set(recipe_ingredients):
            recommended_recipes.append({
                'recipe_id': recipe_ids[idx],
                'recipe_name': recipe_names[idx],
                'cooking_instructions': cooking_instructions[idx],
                'net_price': net_price[idx],
                'image_path': image_paths[idx]  # Include the image path in the recommendation
            })
        if len(recommended_recipes) >= k:
            break

    return recommended_recipes


import json
import numpy as np  # Import numpy for working with int64 data types

def generateIngredients(input_ingredients):
    input_ingredients = [ingredient.strip() for ingredient in input_ingredients.split()]

    # Step 5: Recommend recipes based on the input ingredients
    result = recommend_recipes(input_ingredients)

    res = []

    if result:
        for recipe in result:
            # Convert int64 to int if needed
            net_price = int(recipe['Net Price']) if isinstance(recipe['Net Price'], np.int64) else recipe['Net Price']

            data = {
                "recipe_name": recipe['recipe_name'],
                "cooking_instruction": recipe['cooking_instructions'],
                "net_price": net_price,
                "image_path": recipe["image_path"]
            }
            res.append(data)

        return res  # Serialize the result to JSON
    else:
        return []  # Return an empty JSON array

# Rest of your code...
def get_file_name_from_path(file_path):
    """
    Extracts the file name from a given file path without directories.

    Args:
    file_path (str): The full file path.

    Returns:
    str: The file name (including the extension).
    """
    return os.path.basename(file_path)


from flask import Flask, jsonify,request,send_from_directory

app = Flask(__name__)


@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory('Recipes Pictures', get_file_name_from_path(filename))

@app.route('/recommend_recipes', methods=['POST'])
def api_recommend_recipes():
    data = request.get_json()  # Get JSON data from the request
    message = "None"
    if data:
        res = data['ingredients']
        res = generateIngredients(res)
        message = "success"
    else:
        message = "please put ingredients in request body"
        res = []

    response = {
        "message":message,
        "response":res
    }

    return jsonify(response)



if __name__ == '__main__':
    app.run(debug=True)


