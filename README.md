# Medicine_Recommend_system
📝 Description
This is a Streamlit-based web application that predicts potential diseases based on user-reported symptoms and provides comprehensive recommendations including medications, precautions, diets, and workout plans.

🚀 Features
Symptom Analysis: Users can select multiple symptoms from a comprehensive list

Disease Prediction: Machine learning model predicts the most likely disease

Comprehensive Recommendations:

Detailed disease description

List of recommended precautions

Suggested medications

Appropriate diet plans

Recommended workout routines

User-Friendly Interface: Clean, intuitive design with responsive layout

🛠️ Technical Implementation
Machine Learning: Uses a trained SVC (Support Vector Classifier) model

Data Processing: Pandas for data handling and manipulation

Web Framework: Streamlit for the interactive web interface

Caching: Efficient data and model loading with Streamlit caching

📂 File Structure
text
medicine-recommendation/
├── app.py                  # Main Streamlit application
├── svc.pkl                 # Trained machine learning model
├── precautions_df.csv      # Precautions data
├── workout_df.csv          # Workout recommendations
├── description.csv         # Disease descriptions
├── medications.csv         # Medication information
├── diets.csv               # Diet recommendations
└── medical_image.jpg       # Display image for the app
⚙️ Setup Instructions
Clone the repository:


Disease symptoms and mappings

Precautions for each disease

Recommended medications

Appropriate diets

Suggested workout plans

⚠️ Important Notes
This is for educational/demonstration purposes only

Always consult with a healthcare professional for medical advice

The accuracy depends on the quality of the training data

🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements.

📜 License
MIT License


