from sentence_transformers import SentenceTransformer, util

# Charger le modèle BERT pré-entraîné pour la similarité de phrases
model = SentenceTransformer('bert-base-nli-mean-tokens')

# Fonction pour comparer deux textes avec BERT embeddings
def detect_plagiarism_bert(text1, text2):
    embeddings = model.encode([text1, text2])
    similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1])
    return similarity.item()  # Renvoie la valeur de la similarité

# Exemples de textes
text1 = "Le réchauffement climatique est un problème mondial qui affecte l'ensemble de la planète. Les températures augmentent, les glaciers fondent, et les conditions météorologiques extrêmes deviennent de plus en plus fréquentes. Il est essentiel que nous agissions dès maintenant pour réduire les émissions de gaz à effet de serre et adopter des pratiques plus durables."
text2 = "Les changements climatiques représentent un défi global qui touche toutes les régions du monde. La hausse des températures provoque la fonte des glaciers, et les phénomènes météorologiques extrêmes sont de plus en plus courants. Il est crucial que nous prenions des mesures immédiates pour limiter les émissions de carbone et promouvoir des modes de vie plus respectueux de l'environnement.."

similarity_score = detect_plagiarism_bert(text1, text2)
print(f"Score de similarité (BERT) : {similarity_score}")
