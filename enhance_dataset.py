import pandas as pd

# Load original dataset
df = pd.read_csv("aiml_training_data.csv")

# Realistic customer examples
new_data = {

    "order_status": [
        "Where can I check my order status?",
        "Can you tell me where my order is?",
        "Has my order been shipped?",
        "I want to track my package",
        "What is the current status of my order?",
        "Can I get an update on my order?",
        "Is my package on the way?",
        "Please provide my order tracking information",
        "When was my order shipped?",
        "I need to know the status of my purchase",
        "Where is my package right now?",
        "Can you check my shipment status?",
        "My order tracking hasn't updated",
        "I would like to track my delivery",
        "Has my package left the warehouse?",
        "Could you give me an update about my order?",
        "How do I track my order?",
        "I haven't received any shipping update",
        "Please check where my order is",
        "What is happening with my shipment?",
        "Can I see my order tracking details?",
        "Is my order dispatched yet?",
        "I want an update regarding my purchase",
        "Please tell me the current location of my package",
        "How can I find my order status?"
    ],

    "return_request": [
        "I want to return the product I received",
        "How can I return an item?",
        "I need to send this product back",
        "I would like to request a return",
        "Can I return this item?",
        "I want to return my purchase",
        "What is the process for returning an item?",
        "Please help me initiate a return",
        "I received the product but want to return it",
        "How do I send an item back?",
        "I need to return something I ordered",
        "Can you arrange a return for my product?",
        "I changed my mind and want to return this",
        "I would like to request a product return",
        "Where can I submit a return request?",
        "Please tell me how to return my order",
        "I don't want this product anymore and want to return it",
        "Can this item be returned?",
        "I need help with returning my purchase",
        "What are the steps to return a product?",
        "I want to send my order back",
        "How long do I have to return this item?",
        "Please initiate a return for my order",
        "Can you help me with the return process?",
        "I need to make a return"
    ],

    "refund_enquiry": [
        "When will I receive my refund?",
        "How can I get a refund?",
        "I haven't received my refund yet",
        "Can you tell me the status of my refund?",
        "How long does a refund take?",
        "I want to know about my refund",
        "Please check my refund status",
        "When will the money be returned to me?",
        "Has my refund been processed?",
        "I am waiting for my refund",
        "How do I request a refund?",
        "Why haven't I received my refund?",
        "Can you tell me when my refund will arrive?",
        "I need information about my refund",
        "My refund is still pending",
        "Please help me with my refund",
        "When should I expect the refunded amount?",
        "Was my refund processed successfully?",
        "I want an update on my refund",
        "How long will it take for the refund to reach me?",
        "Can I get a refund for my purchase?",
        "Where is my refunded money?",
        "Please tell me the refund procedure",
        "My refund has not been credited",
        "I need help regarding a refund"
    ],

    "delivery_delay": [
        "My delivery is late",
        "My package hasn't arrived yet",
        "Why is my delivery delayed?",
        "My order is taking too long to arrive",
        "The delivery is overdue",
        "My package is delayed",
        "Why hasn't my order arrived?",
        "My delivery was supposed to arrive earlier",
        "The expected delivery date has passed",
        "My order hasn't been delivered yet",
        "Can you explain why my delivery is late?",
        "My shipment is taking longer than expected",
        "The package has not arrived on time",
        "I am still waiting for my delivery",
        "Why is my package taking so long?",
        "My order is overdue",
        "The delivery date has passed but I haven't received it",
        "Can you check why my order is delayed?",
        "My package is stuck and hasn't arrived",
        "I expected my order by now",
        "There is a delay with my shipment",
        "My delivery hasn't arrived as promised",
        "When will my delayed package arrive?",
        "My order delivery is taking too long",
        "Please help with my delayed delivery"
    ],

    "product_query": [
        "Can you tell me more about this product?",
        "What features does this product have?",
        "I need more information about this item",
        "What are the specifications of this product?",
        "Is this product available?",
        "Can you provide details about this item?",
        "What sizes are available for this product?",
        "What colors does this product come in?",
        "Can you explain the features of this item?",
        "I want to know more about this product",
        "Does this product come in different variants?",
        "What is included with this product?",
        "Can you give me product details?",
        "I have a question about this item",
        "What are the available options for this product?",
        "Tell me about the specifications of this item",
        "Is there more information available about this product?",
        "What is this product used for?",
        "Can you tell me the dimensions of this product?",
        "I would like to know more about the item",
        "What materials is this product made from?",
        "Does this item have different models?",
        "Can you provide more details about the product?",
        "What are the features and specifications?",
        "I have a question regarding this product"
    ],

    "complaint": [
        "I am unhappy with the service",
        "I want to make a complaint",
        "I am not satisfied with my order",
        "The product I received was damaged",
        "I received the wrong item",
        "I am very disappointed with my purchase",
        "The service was poor",
        "I want to report a problem with my order",
        "The product quality is not acceptable",
        "I had a bad experience with my order",
        "I am unhappy about the product I received",
        "The item I received is defective",
        "This is not what I expected",
        "I want to report an issue with my purchase",
        "I am dissatisfied with the service",
        "My order arrived in poor condition",
        "I received a damaged item",
        "The wrong product was delivered to me",
        "I have a problem with the service",
        "I would like to file a complaint",
        "The product does not meet my expectations",
        "I am disappointed with the quality",
        "There is an issue with the item I received",
        "Please help resolve my complaint",
        "I am not happy with my order"
    ]
}

# Convert new examples into a DataFrame
rows = []

for intent, messages in new_data.items():
    for message in messages:
        rows.append({
            "text": message,
            "intent": intent,
            "confidence": 1.0
        })

new_df = pd.DataFrame(rows)

# Remove placeholder samples from original data
realistic_original = df[
    ~df["text"].str.startswith("Sample message")
].copy()

# Combine realistic original data with new examples
enhanced_df = pd.concat(
    [realistic_original, new_df],
    ignore_index=True
)

# Remove duplicate texts
enhanced_df = enhanced_df.drop_duplicates(
    subset=["text"]
).reset_index(drop=True)

# Save enhanced dataset
enhanced_df.to_csv(
    "enhanced_training_data.csv",
    index=False
)

print("=" * 60)
print("ENHANCED DATASET CREATED")
print("=" * 60)
print(f"Original rows:          {len(df)}")
print(f"Realistic original:     {len(realistic_original)}")
print(f"New realistic examples: {len(new_df)}")
print(f"Enhanced total:         {len(enhanced_df)}")
print()
print("Intent distribution:")
print(enhanced_df["intent"].value_counts())
print()
print("Saved as: enhanced_training_data.csv")