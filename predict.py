import joblib


# Load trained model
model = joblib.load("shopsmart_intent_classifier.joblib")


print("=" * 60)
print("SHOPSMART RETAIL - CUSTOMER INTENT CLASSIFIER")
print("=" * 60)
print("Type a customer message to classify it.")
print("Type 'exit' to stop.")
print()


while True:
    text = input("Customer: ").strip()

    if text.lower() == "exit":
        print("Exiting classifier.")
        break

    if not text:
        print("Please enter a message.")
        continue

    prediction = model.predict([text])[0]

    probabilities = model.predict_proba([text])[0]
    confidence = probabilities.max()

    print(f"Predicted intent: {prediction}")
    print(f"Confidence: {confidence:.2%}")
    print()