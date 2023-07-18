
from flask import Flask, request, jsonify

from flask_redis import FlaskRedis

import csv

import string

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity

import nltk

nltk.download('stopwords')

import json

import re
nltk.download('punkt')
# nltk.data.path.append('C:\\Users\\agent\\AppData\\Roaming\\nltk_data')

# try:
#     nltk.data.find('tokenizers/punkt')
# except LookupError:






app = Flask(__name__)

app.config['REDIS_URL'] = 'redis://localhost:6379/0'

redis_store = FlaskRedis(app)




# Flask SQLAlchemy configuration

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///products.db'

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False




from flask_sqlalchemy import SQLAlchemy




db = SQLAlchemy(app)





class Product(db.Model):

    # Add a column to represent the product ID

    product_id = db.Column(db.String(200), primary_key=True)

    # Add a column to represent the product description

    description = db.Column(db.Text())




    def __repr__(self):

        return self.product_id





def preprocess_text(text):

    # Remove punctuation

    text = re.sub(r'[^\w\s]', ' ', text)




    # Convert to lowercase

    text = text.lower()




    # Remove stopwords

    stop_words = set(stopwords.words('english'))

    tokens = nltk.word_tokenize(text)

    tokens = [token for token in tokens if token not in stop_words]

    text = " ".join(tokens)




    return text







@app.route('/train', methods=['POST'])
def train_api():
    if request.method == 'POST':
        try:
            # Read the uploaded CSV file
            csv_file = request.files['file']
            csv_reader = csv.reader(csv_file.read().decode('utf-8').splitlines())

            # Preprocessing setup
            stopwords_set = set(stopwords.words('english'))
            translator = str.maketrans('', '', string.punctuation)

            # Iterate over each row in the CSV file
            training_data = []
            for row in csv_reader:
                if len(row) < 2:
                    continue

                product_id = row[0]
                description = row[1]

                # Preprocess the description
                description = preprocess_text(description)

                if not description or description == "description":
                    continue

                # Check if the product_id already exists in the database
                existing_product = Product.query.get(product_id)
                if existing_product:
                    # Update the description for the existing product
                    existing_product.description = description
                else:
                    # Create a new Product instance and save it to the database
                    product = Product(product_id=product_id, description=description)
                    db.session.add(product)

                db.session.commit()
                training_data.append((product_id, description))

            for product_id, description in training_data:
                print(f"Product ID: {product_id}, Description: {description}")

            return jsonify({'message': 'Training data stored in Redis successfully.'}), 200

        except Exception as e:
            return jsonify({'message': str(e)}), 400

    return jsonify({'message': 'Invalid request method.'}), 400






@app.route('/predict', methods=['POST'])
def predict_api():
    if request.method == 'POST':
        try:
            # Get the user input from the request
            payload = request.get_json()
            input_text = payload.get('input_text', '')

            # Preprocess the input text
            input_text = preprocess_text(input_text)

            # Retrieve all product descriptions from Redis
            product_descriptions = redis_store.smembers('product_descriptions')
            product_descriptions = [desc.decode('utf-8') for desc in product_descriptions]

            # Preprocess the product descriptions
            preprocessed_descriptions = [preprocess_text(desc) for desc in product_descriptions]

            # Calculate similarity scores
            corpus = preprocessed_descriptions + [input_text]
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(corpus)
            similarity_scores = cosine_similarity(tfidf_matrix[:-1], tfidf_matrix[-1]).flatten()

            # Sort the products based on similarity scores
            sorted_indices = similarity_scores.argsort()[::-1]
            sorted_products = [preprocessed_descriptions[idx] for idx in sorted_indices]
            sorted_scores = [similarity_scores[idx] for idx in sorted_indices]

            # Retrieve the top similar products with a score of 50 or above
            threshold_score = 0.25
            top_products = []
            count = 0
            for idx, score in enumerate(sorted_scores):
                if score >= threshold_score:
                    product = Product.query.filter_by(description=sorted_products[idx]).first()
                    if product:
                        top_products.append({
                            'product_id': product.product_id,
                            'description': product.description,
                            'similarity_score': score
                        })
                        count += 1
                        if count >= 3:  # Limit to top 3 products
                            break

            return jsonify({'predictions': top_products}), 200

        except Exception as e:
            return jsonify({'message': str(e)}), 400

    return jsonify({'message': 'Invalid request method.'}), 400














if __name__ == '__main__':

    with app.app_context():

        db.create_all()

    app.run(debug=True)


