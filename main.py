import pandas as pd
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import messagebox
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt



# Stałe
CSV_PATH = "credit_data_synthetic.csv"

FEATURES = [
    "wiek",
    "dochod_miesieczny",
    "zadluzenie",
    "historia_kredytowa",
    "stabilnosc_zatrudnienia",
    "kwota_kredytu",
    "okres_kredytu_mies",
    "liczba_zaleglosci",
    "rata_do_dochodu"
]

TARGET = "decyzja"



# Wczytanie danych
df = pd.read_csv(CSV_PATH)
X_data = df[FEATURES]
y = df[TARGET]



# Trenowanie modelu
def train_model(dataframe):
    X = dataframe[FEATURES]
    y = dataframe[TARGET]

    model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=6,
        min_samples_leaf=10,
        random_state=42
    )
    model.fit(X, y)
    return model

model = train_model(df)



# Funkcje do przycisków
def show_tree():
    plt.figure(figsize=(28, 10))
    plot_tree(
        model,
        feature_names=FEATURES,
        class_names=["NIE", "TAK"],
        filled=True
    )
    plt.title("Drzewo decyzyjne – decyzja kredytowa")
    plt.show()


def show_metrics():
    y_pred = model.predict(X_data)
    cm = confusion_matrix(y, y_pred)

    text = (
        f"Accuracy (Dokładność): {accuracy_score(y, y_pred):.2f}\n"
        f"Precision (Precyzja): {precision_score(y, y_pred):.2f}\n"
        f"Recall (Czułość): {recall_score(y, y_pred):.2f}\n"
        f"F1-score (Miara F1): {f1_score(y, y_pred):.2f}\n\n"
        f"Macierz konfuzji:\n"
        f"TN={cm[0,0]}  FP={cm[0,1]}\n"
        f"FN={cm[1,0]}  TP={cm[1,1]}"
    )
    messagebox.showinfo("Jakość modelu", text)


def show_feature_importance():
    importances = model.feature_importances_
    plt.figure(figsize=(10, 6))
    plt.barh(FEATURES, importances)
    plt.xlabel("Ważność cechy")
    plt.title("Ważność cech w modelu")
    plt.tight_layout()
    plt.show()



# Logika decyzji + zapis
def evaluate_and_save():
    global df, model, X_data, y

    try:
        values = [float(entries[row].get()) for row in FEATURES]
        data = pd.DataFrame([values], columns=FEATURES)

        decision = int(model.predict(data)[0])

        messagebox.showinfo(
            "Decyzja kredytowa",
            "PRZYZNAĆ KREDYT" if decision == 1 else "ODRZUCIĆ WNIOSEK"
        )

        # zapis nowego przypadku
        data[TARGET] = decision
        df = pd.concat([df, data], ignore_index=True)
        df.to_csv(CSV_PATH, index=False)

        # ponowne trenowanie modelu na rozszerzonych danych
        X_data = df[FEATURES]
        y = df[TARGET]
        model = train_model(df)

    except Exception as e:
        messagebox.showerror("Błąd", str(e))




# GUI
app = ttk.Window(themename="darkly")
app.title("System wspomagania decyzji kredytowej")
app.geometry("900x650")

#     ekran startowy
start_frame = ttk.Frame(app)
start_frame.pack(expand=True)

ttk.Label(
    start_frame,
    text="System wspomagania decyzji kredytowej",
    font=("Arial", 24, "bold")
).pack(pady=30)

ttk.Button(
    start_frame,
    text="Dodaj wniosek",
    bootstyle=SUCCESS,
    command=lambda: open_form()
).pack(pady=10)

ttk.Button(start_frame, text="Pokaż drzewo decyzyjne", command=show_tree).pack(pady=5)
ttk.Button(start_frame, text="Pokaż metryki", command=show_metrics).pack(pady=5)
ttk.Button(start_frame, text="Ważność cech", command=show_feature_importance).pack(pady=5)


#     formularz
form_frame = ttk.Frame(app)
entries = {}

def clear_form():
    for entry in entries.values():
        entry.delete(0, END)


def go_back_to_menu():
    clear_form()
    form_frame.pack_forget()
    start_frame.pack(expand=True)

def open_form():
    start_frame.pack_forget()
    form_frame.pack(expand=True)

    if entries:
        clear_form()
        return

    ttk.Label(
        form_frame,
        text="Dane klienta",
        font=("Arial", 20, "bold")
    ).pack(pady=10)

    for f in FEATURES:
        row = ttk.Frame(form_frame)
        row.pack(fill=X, pady=2)
        ttk.Label(row, text=f, width=30).pack(side=LEFT)
        e = ttk.Entry(row)
        e.pack(side=LEFT, expand=True, fill=X)
        entries[f] = e

    ttk.Button(
        form_frame,
        text="Zatwierdź",
        bootstyle=SUCCESS,
        command=evaluate_and_save
    ).pack(pady=10)

    ttk.Button(
        form_frame,
        text="Powrót do menu",
        bootstyle=SECONDARY,
        command=go_back_to_menu
    ).pack(pady=5)


app.mainloop()
