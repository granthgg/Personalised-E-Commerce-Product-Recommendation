from flask import Flask, render_template, request, session, redirect, url_for

import pandas as pd
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import networkx as nx
from community import community_louvain
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
app.secret_key = '90308296727364542f482a7cea225999583afa1dd3031793c89b4c5ff19d5966'


# Load data into dataframes
interest = pd.read_csv('interest_scores.csv')
products = pd.read_csv('product.csv')
interactions = pd.read_csv('interactions.csv')
users = pd.read_csv('user.csv')
purchases = pd.read_csv('purchases.csv')


@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/select_user')
def select_user():
    user_ids = users['User ID'].tolist()
    return render_template('select_user.html', user_ids=user_ids)

from collections import defaultdict

@app.route('/product/<int:product_id>')
def product_details(product_id):
    selected_product = products[products['Product ID'] == product_id].iloc[0]
    return render_template('product_details.html', product=selected_product)

@app.route('/browse_products')
def browse_products():
    # Get all products (you need to implement this logic)
    all_products = products  # Placeholder, replace with actual logic

    # Group products by category
    products_by_category = defaultdict(list)
    for product in all_products.to_dict('records'):
        products_by_category[product['Category']].append(product)

    return render_template('browse_products.html', products_by_category=products_by_category)

class Cart:
    def __init__(self):
        self.products = []
        
        session.clear()

        
    def add_product(self, product):
        if 'Quantity' not in product:
            product['Quantity'] = 1
            
        for cart_product in self.products:
            if cart_product['Product ID'] == product['Product ID']:
                cart_product['Quantity'] += 1
                return
        self.products.append(product)


    def clear_cart(self):
        self.products = []

        
    def remove_product(self, product_id):
        product_index = next((index for index, product in enumerate(self.products) if product['Product ID'] == product_id), None)
        
        if product_index is not None:
            self.products.pop(product_index)

        
    def get_products(self):
        return self.products
        
    def get_total_price(self):
        total_price = sum(product['Price'] * product['Quantity'] for product in self.products)
        return total_price
    
    def to_json(self):
        return {'products': self.products}
    
    @classmethod
    def from_json(cls, data):
        cart = cls()
        cart.products = data['products']
        return cart



@app.route('/remove_from_cart/<int:product_id>')
def remove_from_cart(product_id):
    #session.clear()

    # Create or retrieve an instance of the Cart class
    if 'cart' not in session:
        cart = Cart()
    else:
        cart = Cart.from_json(session['cart'])
    
    # Remove the selected product from the cart
    cart.remove_product(product_id)
    
    # Save the updated cart to the session
    session['cart'] = cart.to_json()
    
    # Redirect to the cart page
    return redirect(url_for('cart', product_id=product_id))


@app.route('/add_to_cart/<int:product_id>', methods=['POST'])
def add_to_cart(product_id):
    #session.clear()

    # Get the quantity from the form submission
    quantity = int(request.form['quantity'])
    
    # Create or retrieve an instance of the Cart class
    if 'cart' not in session:
        cart = Cart()
    else:
        cart = Cart.from_json(session['cart'])
    
    # Get the selected product
    selected_product = products[products['Product ID'] == product_id].iloc[0].to_dict()
    
    # Add the specified quantity of the selected product to the cart
    for _ in range(quantity):
        cart.add_product(selected_product)
    
    # Save the updated cart to the session
    session['cart'] = cart.to_json()
    
    # Redirect to the cart page
    return redirect(url_for('cart', product_id=product_id))




@app.route('/cart/<int:product_id>')
def cart(product_id):
    #session.clear()

    # Create or retrieve an instance of the Cart class
    if 'cart' not in session:
        cart = Cart()
    else:
        cart = Cart.from_json(session['cart'])
    
    # Get the selected product
    selected_product = products[products['Product ID'] == product_id].iloc[0].to_dict()
    
    # Add the selected product to the cart
    cart.add_product(selected_product)
    
    # Save the updated cart to the session
    session['cart'] = cart.to_json()
    
    # Get the current contents of the cart
    cart_products = cart.get_products()
    
    # Get the total price of the cart
    total_price = cart.get_total_price()
    
    return render_template('cart.html', cart_products=cart_products, total_price=total_price)

@app.route('/confirmation')
def confirmation():
    if 'cart' not in session:
        cart = Cart()
    else:
        cart = Cart.from_json(session['cart'])
    
    # Get the purchased products from the cart
    purchased_products = cart.get_products()
    
    # Get the total price of the cart
    total_price = cart.get_total_price()
    
    # Clear the cart after the purchase is confirmed
    cart.clear_cart()
    
    return render_template('confirmation.html', purchased_products=purchased_products, total_price=total_price)


@app.route('/user_info', methods=['POST'])
def user_info():
    user_id = int(request.form['user_id'])
    user_details = users[users['User ID'] == user_id].iloc[0]
    return render_template('user_info.html', user_id=user_details['User ID'],
                           user_age=user_details['Age'], user_gender=user_details['Gender'],
                           user_location=user_details['Location'],
                           subscription_status=user_details['Subscription Status'])

from datetime import datetime

@app.route('/interaction_history/<int:user_id>')
def interaction_history(user_id):
    user_viewed_interactions = interactions[(interactions['User ID'] == user_id) & (interactions['Interaction Type'] == 'Viewed')]
    user_added_to_cart_interactions = interactions[(interactions['User ID'] == user_id) & (interactions['Interaction Type'] == 'Added to Cart')]
    user_purchased_interactions = interactions[(interactions['User ID'] == user_id) & (interactions['Interaction Type'] == 'Purchased')]

    # Fetch user's purchased interactions and ratings
    purchased_interactions = purchases[purchases['User ID'] == user_id]
    
    # Merge purchased_interactions with products to get product details
    purchased_interactions = pd.merge(purchased_interactions, products, on='Product ID')

    # Convert the timestamps into a more readable format
    user_viewed_interactions['Interaction Timestamp'] = user_viewed_interactions['Interaction Timestamp'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S').strftime('%b %d, %Y'))
    user_added_to_cart_interactions['Interaction Timestamp'] = user_added_to_cart_interactions['Interaction Timestamp'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S').strftime('%b %d, %Y'))
    purchased_interactions['Purchase Timestamp'] = purchased_interactions['Purchase Timestamp'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S').strftime('%b %d, %Y'))

    return render_template('interaction_history.html',
                           viewed_interactions=user_viewed_interactions.to_dict('records'),
                           added_to_cart_interactions=user_added_to_cart_interactions.to_dict('records'),
                           purchased_interactions=purchased_interactions.to_dict('records'))





@app.route('/recommended_products/<int:user_id>')
def recommended_products(user_id):
    interest_scores_df = pd.read_csv('interest_scores.csv')
    product_df = pd.read_csv('product.csv')
    purchase_df = pd.read_csv('purchases.csv')
    interactions_df = pd.read_csv('interactions.csv')
    user_df = pd.read_csv('user.csv')

    purchase_df = purchase_df.drop_duplicates(subset=['User ID', 'Product ID'])

    interest_scores = interest_scores_df.set_index('User ID').apply(pd.to_numeric, errors='coerce').fillna(0)

    scaler = StandardScaler()
    normalized_interest_scores = scaler.fit_transform(interest_scores)

    user_item_matrix = purchase_df.pivot(index='User ID', columns='Product ID', values='Rating').fillna(0)

    collab_model = NearestNeighbors(metric='cosine', algorithm='brute')
    collab_model.fit(user_item_matrix)

    product_category_matrix = product_df.set_index('Product ID')['Category']

    user_interests = normalized_interest_scores[user_id]

    if user_interests.size > 0:
        top_interest_category = interest_scores_df.columns[user_interests.argmax()]

        top_interest_products = product_df[product_df['Category'] == top_interest_category]['Product ID']

        content_recommendations = top_interest_products.values
    else:
        content_recommendations = []

    G = nx.Graph()
    for _, row in interactions_df.iterrows():
        G.add_edge(row['User ID'], row['Product ID'])

    partition = community_louvain.best_partition(G)

    user_community = partition[user_id]

    community_products = [node for node, comm in partition.items() if comm == user_community]

    X = []  # Feature matrix
    y = []  # Target vector

    purchased_products = set(purchase_df[purchase_df['User ID'] == user_id]['Product ID'])
    relevant_products = purchased_products  # Use purchased products as relevant products

    distances, indices = collab_model.kneighbors(user_item_matrix.iloc[user_id].values.reshape(1, -1), n_neighbors=6)
    collab_recommendations = user_item_matrix.iloc[indices.flatten()].index

    for product_id in set(collab_recommendations) | set(content_recommendations) | set(community_products):
        features = [0, 0, 0]
        if product_id in collab_recommendations:
            features[0] = 1
        if product_id in content_recommendations:
            features[1] = 1
        if product_id in community_products:
            features[2] = 1
        X.append(features)

        if product_id in relevant_products:
            y.append(1)
        else:
            y.append(0)

    model = LinearRegression()
    model.fit(X, y)


    combined_recommendations = []
    for product_id in set(collab_recommendations) | set(content_recommendations) | set(community_products):
        features = [0, 0, 0]
        if product_id in collab_recommendations:
            features[0] = 1
        if product_id in content_recommendations:
            features[1] = 1
        if product_id in community_products:
            features[2] = 1
        score = model.predict([features])[0]
        combined_recommendations.append((product_id, score))

    N = 10


    combined_recommendations.sort(key=lambda x: x[1], reverse=True)
    top_recommendations = [item[0] for item in combined_recommendations[:N]]

    recommended_products_df = product_df[product_df['Product ID'].isin(top_recommendations)]
    
    return render_template('recommended_products.html', user_id=user_id, products=recommended_products_df.to_dict('records'))

if __name__ == '__main__':
    app.run(debug=True)