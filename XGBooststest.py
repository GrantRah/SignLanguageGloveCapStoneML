import joblib
import numpy as np
import pandas as pd
import os

class GesturePredictor:
    def __init__(self, model_path='xgboost_gesture_model.pkl'):
        """
        Initialize the gesture predictor with the trained model
        
        Args:
            model_path: Path to the saved model file
        """
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_columns = None
        self.class_names = None
        
        self.load_model()
    
    def load_model(self):
        """Load the trained model and preprocessors"""
        if not os.path.exists(self.model_path):
            print(f"❌ Model file '{self.model_path}' not found!")
            print("Please run the training script first to create the model.")
            return False
        
        try:
            # Load the model artifacts
            artifacts = joblib.load(self.model_path)
            self.model = artifacts['model']
            self.scaler = artifacts['scaler']
            self.label_encoder = artifacts['label_encoder']
            self.feature_columns = artifacts['feature_columns']
            self.class_names = artifacts.get('class_names', 
                                            [str(c) for c in self.label_encoder.classes_])
            
            print("✅ Model loaded successfully!")
            print(f"   Model accuracy: {artifacts.get('accuracy', 'N/A'):.4f}" if 'accuracy' in artifacts else "   Model loaded")
            print(f"   Number of classes: {len(self.class_names)}")
            print(f"   Classes: {', '.join(self.class_names)}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def predict_from_values(self, flex_values):
        """
        Predict gesture from flex sensor values
        
        Args:
            flex_values: List or array of 5 flex sensor values [Flex1, Flex2, Flex3, Flex4, Flex5]
        
        Returns:
            tuple: (predicted_class, confidence, all_probabilities)
        """
        # Validate input
        if len(flex_values) != 5:
            raise ValueError(f"Expected 5 flex sensor values, got {len(flex_values)}")
        
        # Convert to numpy array and reshape for prediction
        features = np.array(flex_values).reshape(1, -1)
        
        # Scale the features
        features_scaled = self.scaler.transform(features)
        
        # Get prediction
        prediction_encoded = self.model.predict(features_scaled)[0]
        prediction_proba = self.model.predict_proba(features_scaled)[0]
        
        # Get class name and confidence
        predicted_class = self.class_names[prediction_encoded]
        confidence = np.max(prediction_proba)
        
        # Create dictionary of all class probabilities
        all_probabilities = {
            class_name: prob for class_name, prob in zip(self.class_names, prediction_proba)
        }
        
        return predicted_class, confidence, all_probabilities
    
    def predict_from_csv_row(self, row_data):
        """
        Predict gesture from a CSV row (dictionary or pandas Series)
        
        Args:
            row_data: Dictionary or Series containing Flex1-Flex5 values
        
        Returns:
            tuple: (predicted_class, confidence, all_probabilities)
        """
        flex_values = [
            row_data['Flex1'], row_data['Flex2'], 
            row_data['Flex3'], row_data['Flex4'], row_data['Flex5']
        ]
        return self.predict_from_values(flex_values)
    
    def interactive_mode(self):
        """Run interactive console mode for real-time predictions"""
        print("\n" + "="*60)
        print("🖐️  Gesture Recognition System - Interactive Mode")
        print("="*60)
        print("\nEnter the 5 flex sensor values (0-4095 typical range)")
        print("You can also type 'help' for information or 'quit' to exit\n")
        
        while True:
            try:
                # Get user input
                user_input = input("\n📊 Enter Flex1 Flex2 Flex3 Flex4 Flex5 (space-separated): ").strip()
                
                # Check for exit commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                
                # Check for help
                if user_input.lower() == 'help':
                    print("\n📖 Help:")
                    print("  - Enter 5 numbers separated by spaces")
                    print("  - Example: 2680 2690 2570 2580 2400")
                    print("  - Values typically range from 0-4095")
                    print("  - Type 'quit' to exit")
                    print("  - Type 'example' to see an example")
                    continue
                
                # Check for example
                if user_input.lower() == 'example':
                    print("\n📋 Example input (using sample from training data):")
                    print("  Flex1=2680, Flex2=2690, Flex3=2570, Flex4=2580, Flex5=2400")
                    print("  Enter: 2680 2690 2570 2580 2400")
                    continue
                
                # Parse input
                values = user_input.split()
                if len(values) != 5:
                    print(f"❌ Please enter exactly 5 values (you entered {len(values)})")
                    continue
                
                # Convert to float/int
                flex_values = []
                for i, val in enumerate(values):
                    try:
                        flex_values.append(float(val))
                    except ValueError:
                        print(f"❌ Invalid number: '{val}' at position {i+1}")
                        break
                else:
                    # All values are valid, make prediction
                    predicted_class, confidence, all_probs = self.predict_from_values(flex_values)
                    
                    # Display results
                    print("\n" + "="*60)
                    print("🎯 Prediction Result:")
                    print("="*60)
                    print(f"  Predicted Gesture: {predicted_class}")
                    print(f"  Confidence: {confidence:.2%}")
                    
                    # Show top 3 predictions
                    print("\n  Top predictions:")
                    sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
                    for i, (gesture, prob) in enumerate(sorted_probs[:3], 1):
                        bar = "█" * int(prob * 40)
                        print(f"    {i}. {gesture}: {prob:.2%} {bar}")
                    
                    # Input values
                    print("\n  Input values:")
                    for i, (col, val) in enumerate(zip(['Flex1', 'Flex2', 'Flex3', 'Flex4', 'Flex5'], flex_values)):
                        print(f"    {col}: {val}")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                print("Please try again or type 'help' for assistance")
    
    def batch_predict_from_csv(self, csv_file_path, output_file=None):
        """
        Predict gestures for all rows in a CSV file
        
        Args:
            csv_file_path: Path to CSV file with Flex1-Flex5 columns
            output_file: Path to save results (optional)
        
        Returns:
            DataFrame with predictions added
        """
        try:
            # Load CSV
            df = pd.read_csv(csv_file_path)
            print(f"✅ Loaded {len(df)} rows from {csv_file_path}")
            
            # Check required columns
            required_cols = ['Flex1', 'Flex2', 'Flex3', 'Flex4', 'Flex5']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"❌ Missing columns: {missing_cols}")
                return None
            
            # Make predictions for each row
            predictions = []
            confidences = []
            
            for idx, row in df.iterrows():
                try:
                    flex_values = [row['Flex1'], row['Flex2'], row['Flex3'], row['Flex4'], row['Flex5']]
                    pred_class, confidence, _ = self.predict_from_values(flex_values)
                    predictions.append(pred_class)
                    confidences.append(confidence)
                except Exception as e:
                    print(f"⚠️ Error predicting row {idx}: {e}")
                    predictions.append('ERROR')
                    confidences.append(0)
            
            # Add predictions to dataframe
            df['Predicted_Sign'] = predictions
            df['Confidence'] = confidences
            
            # Save if output file specified
            if output_file:
                df.to_csv(output_file, index=False)
                print(f"✅ Results saved to {output_file}")
            
            # Display summary
            print("\n📊 Prediction Summary:")
            print(f"  Total rows: {len(df)}")
            print(f"  Successful predictions: {(df['Predicted_Sign'] != 'ERROR').sum()}")
            
            # Show distribution of predictions
            if (df['Predicted_Sign'] != 'ERROR').any():
                print("\n  Predicted class distribution:")
                pred_counts = df[df['Predicted_Sign'] != 'ERROR']['Predicted_Sign'].value_counts()
                for gesture, count in pred_counts.head(10).items():
                    print(f"    {gesture}: {count} ({count/len(df)*100:.1f}%)")
            
            return df
            
        except Exception as e:
            print(f"❌ Error processing CSV: {e}")
            return None


def main():
    """Main function to run the gesture prediction system"""
    
    print("="*60)
    print("🖐️  Sign Language Glove - Gesture Prediction System")
    print("="*60)
    
    # Initialize predictor
    predictor = GesturePredictor('xgboost_gesture_model.pkl')
    
    # Check if model loaded successfully
    if not predictor.model:
        print("\n⚠️ Model not found. Please run the training script first:")
        print("  python XGBGTthatworks.py")
        return
    
    # Menu system
    while True:
        print("\n" + "="*60)
        print("Main Menu:")
        print("="*60)
        print("1. 🔮 Interactive Mode (predict single gesture)")
        print("2. 📁 Batch Process CSV File")
        print("3. 📊 View Model Information")
        print("4. 🧪 Run Test Predictions")
        print("5. ❌ Exit")
        
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == '1':
            predictor.interactive_mode()
        
        elif choice == '2':
            csv_path = input("Enter path to CSV file: ").strip()
            if os.path.exists(csv_path):
                output_path = input("Enter output file path (optional, press Enter to skip): ").strip()
                if not output_path:
                    output_path = None
                predictor.batch_predict_from_csv(csv_path, output_path)
            else:
                print(f"❌ File not found: {csv_path}")
        
        elif choice == '3':
            print("\n📊 Model Information:")
            print("="*60)
            print(f"  Model file: {predictor.model_path}")
            print(f"  Number of classes: {len(predictor.class_names)}")
            print(f"  Classes: {', '.join(predictor.class_names)}")
            print(f"  Features used: {', '.join(predictor.feature_columns)}")
            print(f"  Input range: Typical flex sensor values 0-4095")
        
        elif choice == '4':
            print("\n🧪 Running test predictions with sample data...")
            # Test with some sample values from training
            test_samples = [
                [2680, 2690, 2570, 2580, 2400],  # Sample from class 1
                [2690, 2740, 2970, 2600, 2460],  # Sample from class 2
                [2900, 2740, 2960, 2620, 2450],  # Sample from class 3
                [2470, 2750, 2970, 2950, 2800],  # Sample from class 4
            ]
            
            for i, sample in enumerate(test_samples, 1):
                pred_class, confidence, _ = predictor.predict_from_values(sample)
                print(f"\n  Test {i}: {sample}")
                print(f"    → Predicted: {pred_class} (confidence: {confidence:.2%})")
        
        elif choice == '5':
            print("\n👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please select 1-5")


if __name__ == "__main__":
    # Simple single prediction mode (if called directly with arguments)
    import sys
    
    if len(sys.argv) == 6:
        # Command line mode: python predict_gesture.py 2680 2690 2570 2580 2400
        try:
            flex_values = [float(x) for x in sys.argv[1:6]]
            predictor = GesturePredictor()
            if predictor.model:
                pred_class, confidence, probs = predictor.predict_from_values(flex_values)
                print(f"\nPredicted Gesture: {pred_class}")
                print(f"Confidence: {confidence:.2%}")
                
                # Show top 3
                print("\nTop 3 predictions:")
                sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
                for gesture, prob in sorted_probs[:3]:
                    print(f"  {gesture}: {prob:.2%}")
        except Exception as e:
            print(f"Error: {e}")
            print("Usage: python predict_gesture.py flex1 flex2 flex3 flex4 flex5")
    else:
        # Interactive mode
        main()