# Chess Game Predictor

A machine learning project designed to analyze and predict outcomes of chess games using historical grandmaster data.

## Features
- Parses standard PGN (Portable Game Notation) files.
- Feature engineering based on material balance, piece mobility, and king safety.
- Machine learning model to predict Win/Loss/Draw probabilities.

## Project Structure
```text
├── data/
│   └── professional/    # (Local only) Store downloaded PGN files here
├── models/              # Saved model weights (.pkl or .h5)
├── notebooks/           # Jupyter notebooks for EDA
├── src/                 # Source code for processing and training
├── .gitignore           # Keeps the repo lean (ignores datasets/JDK)
└── README.md
````

## Data Setup (Required)

To keep the repository lightweight, the raw dataset is **not** included in this repository. Please follow these steps to set up the data:

1.  Visit [PGNMentor](https://www.pgnmentor.com/files.html).
2.  Download the desired player or event archives (e.g., "Kasparov.pgn" or "WorldChampionship.pgn").
3.  Create a folder named `data/professional/` in the root of this project.
4.  Extract/Move the `.pgn` files into that folder.

## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/pradeepmalar/chess_game_pred.git
    cd chess_game_pred
    ```

2.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the main processing script to start training the model:

```bash
python src/main.py
```

## License

Distributed under the MIT License. See `LICENSE` for more information.

-----

**Author:** [Pradeep Malar](https://www.google.com/search?q=https://github.com/pradeepmalar)

````
