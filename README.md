# Personalised E-Commerce Product Recommendation System

## Overview
This project aims to create a personalized product recommendation system for e-commerce platforms. It leverages user interaction data, purchase history, and product information to generate tailored recommendations, enhancing the shopping experience and potentially increasing sales.

## Features
- **Data Exploration and Preprocessing**: Jupyter notebooks for exploring, cleaning, and preparing the dataset for the recommendation algorithm.
- **Recommendation Algorithm**: Implementation of an interest calculation algorithm to score user interest in products based on past interactions and purchases.
- **Web Application**: A Flask web application to showcase the recommendation system in action, allowing users to browse products, view recommendations, and simulate purchases.

## Project Structure
- `Code For Recommendation System/`: Contains all the Jupyter notebooks and CSV files related to the recommendation algorithm.
  - `Data Exploration.ipynb`: Notebook for initial data exploration.
  - `Data Generation.ipynb`: Notebook for generating synthetic data for testing.
  - `Interest Calculation-Algorithm.ipynb`: Notebook detailing the algorithm for calculating user interest scores.
  - `Recommendation.ipynb`: Notebook for generating product recommendations.
  - CSV files: Include `user.csv`, `product.csv`, `purchases.csv`, `interactions.csv`, `interest_scores.csv`, and `all_recommendations.csv` for algorithm inputs and outputs.
- `Code for Website to showcase Recommendation System/`: Contains the Flask application and static assets.
  - `app.py`: The Flask application.
  - `templates/`: HTML templates for the web interface.
  - `static/`: CSS and image files for the web application.
  - CSV files: Same as above, used by the web application to display recommendations.
- `Proof of Concept PPT.pdf`: A presentation outlining the concept and implementation of the project.
- `Report-Personalised Product Recommendation System.pdf`: A detailed report on the project, including methodology, results, and conclusions.

## Getting Started
To run the recommendation system:
1. Clone the repository.
2. Navigate to the `Code For Recommendation System/` directory to explore the Jupyter notebooks.
3. To run the web application:
   - Ensure you have Python and Flask installed.
   - Navigate to the `Code for Website to showcase Recommendation System/` directory.
   - Run `app.py` using Flask.
   - Access the web application at `http://localhost:5000`.

## Technologies Used
- Python: For data processing and the web server.
- Jupyter Notebook: For data exploration and algorithm development.
- Flask: For the web application.
- HTML/CSS: For the web interface.

## Contributing
Contributions to improve the recommendation system or the web application are welcome. Please fork the repository, make your changes, and submit a pull request.

