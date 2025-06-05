content-sorter/
├── main.py                  # CLI-hantering och entry point
├── config.py                # Inställningar och konfigurationer
├── categorizer/
│   ├── __init__.py
│   ├── loader.py            # Filuppläsning & textutvinning
│   ├── embedder.py          # Embedding av filinnehåll
│   ├── clusterer.py         # Klustring av embeddings
│   ├── suggester.py         # Fråga LLM om kategorinamn
│   ├── organizer.py         # Flytta filer till kategorimappar
├── models/
│   ├── llm_interface.py     # Wrapper mot lokal 7B-modell (Ollama/HF)
│   └── __init__.py
├── utils/
│   ├── logger.py            # Färggrann loggning
│   └── file_utils.py        # Hjälpfunktioner för filsystemet
├── requirements.txt         # Pythonberoenden
└── README.md                # Dokumentation