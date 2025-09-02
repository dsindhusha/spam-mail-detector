import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset and train model
@st.cache_resource
def load_data_and_train():
    df = pd.read_csv(
        "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv",
        sep="\t",
        header=None,
        names=["label", "message"]
    )
    df["label"] = df["label"].map({"ham": 0, "spam": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        df["message"], df["label"], test_size=0.2, random_state=42
    )

    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    return model, vectorizer, accuracy, cm, report

model, vectorizer, accuracy, cm, report = load_data_and_train()

# Streamlit UI
st.set_page_config(page_title="Spam Mail Detector", page_icon="📧", layout="centered")
st.title("📧 Spam Mail Detection App")
st.write("This app uses a **Naive Bayes Classifier** to detect whether a message is Spam or Ham.")

# Sidebar - Performance
st.sidebar.header("Model Performance")
st.sidebar.metric("Accuracy", f"{accuracy*100:.2f}%")
st.sidebar.write("### Confusion Matrix")
st.sidebar.dataframe(pd.DataFrame(cm, index=["Ham (0)", "Spam (1)"], columns=["Pred Ham", "Pred Spam"]))
st.sidebar.write("### Precision/Recall/F1")
st.sidebar.dataframe(pd.DataFrame(report).transpose())

# User Input
user_input = st.text_area("✍️ Enter a message to classify:", "")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)[0]
        label = "🚫 Spam" if prediction == 1 else "✅ Ham"
        st.success(f"Prediction: **{label}**")
