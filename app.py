import streamlit as st
import pickle
from rdkit import Chem
from rdkit.Chem import Descriptors

# Load your trained model
with open("toxicity_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("?? Drug Toxicity Prediction App")

st.markdown("""
Enter the **SMILES string** of the drug molecule below to predict whether it is **Safe** or **Toxic**.
""")

# Function to convert SMILES to features
def featurize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    # Features used in your trained model (example: replace if needed)
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    return [mw, logp, hbd, hba]  # Make sure this matches your model

# Single prediction input
smiles = st.text_input("Enter Drug SMILES:")

if st.button("Predict"):
    if smiles:
        features = featurize(smiles)
        if features is None:
            st.error("? Invalid SMILES string")
        else:
            prediction = model.predict([features])[0]
            probability = model.predict_proba([features])[0][1]  # Optional probability
            result = "Toxic ??" if prediction else "Safe ?"
            st.success(f"Prediction: {result}")
            st.info(f"Toxicity Probability: {probability:.2f}")
    else:
        st.warning("?? Please enter a SMILES string")

# Optional: Batch prediction (CSV upload)
uploaded_file = st.file_uploader("Or upload CSV (with column 'SMILES')", type=["csv"])
if uploaded_file:
    import pandas as pd
    data = pd.read_csv(uploaded_file)
    if "SMILES" not in data.columns:
        st.error("CSV must have a 'SMILES' column")
    else:
        predictions = []
        probabilities = []
        for smi in data["SMILES"]:
            features = featurize(smi)
            if features is None:
                predictions.append("Invalid")
                probabilities.append(0)
            else:
                pred = model.predict([features])[0]
                prob = model.predict_proba([features])[0][1]
                predictions.append("Toxic ??" if pred else "Safe ?")
                probabilities.append(prob)
        data["Prediction"] = predictions
        data["Probability"] = probabilities
        st.dataframe(data)
        st.download_button("Download Results", data.to_csv(index=False), file_name="predictions.csv")