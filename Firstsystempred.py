"""
explaniation:
User loads the trained model and inputs features to get quality predictions
This is the part that will be integrated into the app once its ready <3
"""
import joblib
import numpy as np

class WineQualityPredictor:
    def __init__(self, model_path='wine_model.joblib', features_path='feature_names.joblib'):
        """Load trained model and feature information"""
        self.model = joblib.load(model_path)
        self.feature_names = joblib.load(features_path)
        print(f"✓ Model loaded successfully")
        print(f"✓ Features: {self.feature_names}")
        print(f"✓ Ready for predictions!\n")
    
    def predict_from_input(self):
        """Get input from user and make prediction"""
        print("="*50)
        print("WINE QUALITY PREDICTOR")
        print("="*50)
        print("Enter the following wine characteristics:\n")
        
        features = []
        for feature in self.feature_names:
            # Convert feature name to user-friendly format
            friendly_name = feature.replace('_', ' ').title()
            value = float(input(f"{friendly_name}: "))
            features.append(value)
        
        # Convert to numpy array and reshape for prediction
        features_array = np.array(features).reshape(1, -1)
        
        # Make prediction
        prediction = self.model.predict(features_array)[0]
        probabilities = self.model.predict_proba(features_array)[0]
        
        # Display results
        print("\n" + "="*50)
        print("PREDICTION RESULTS")
        print("="*50)
        print(f"Predicted Wine Quality: {prediction}/10")
        
        # Show probabilities for all possible quality scores
        quality_scores = sorted(self.model.classes_)
        print("\nProbability Distribution:")
        for score, prob in zip(quality_scores, probabilities):
            print(f"  Quality {score}: {prob*100:.1f}%")
        
        # Get confidence level
        max_prob = max(probabilities) * 100
        if max_prob > 70:
            confidence = "High"
        elif max_prob > 50:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        print(f"\nConfidence: {confidence} ({max_prob:.1f}%)")
        print("="*50 + "\n")
        
        return prediction, features

if __name__ == "__main__":
    # Create predictor
    predictor = WineQualityPredictor()
    
    # Run prediction loop
    while True:
        predictor.predict_from_input()
        
        # Ask if user wants to continue
        continue_pred = input("Make another prediction? (y/n): ").lower()
        if continue_pred != 'y':
            print("Goodbye!")
            break