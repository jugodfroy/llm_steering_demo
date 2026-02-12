# Resultats des tests de steering par paires contrastives

## Resume

Extraction de vecteurs de steering via paires contrastives (RepE) sur Llama 3.1 8B Instruct.
8 concepts testes, 3 couches (12, 15, 19), forces 4-16.

---

## Concepts qui fonctionnent bien

### PIRATE - Effet WOW (excellent)

| Config | Qualite | Commentaire |
|--------|---------|-------------|
| L12 S4 | OK | Style legerement plus dramatique, pas vraiment pirate |
| L12 S8 | Degrade | Devient poetique/archaique, perd la coherence |
| L15 S4 | Bon | Commence a montrer du langage pirate ("matey!") tout en restant coherent |
| **L15 S8** | **Excellent** | **Full pirate speak, "Arrr matey!", coherent et amusant** |
| L15 S12 | OK | Pirate mais commence a devenir repetitif |
| L15 S16 | Degrade | Charabia |
| L19 S8 | Bon | Pirate speech clair et coherent |

**Config recommandee: Layer 15, Strength 8** - Effet tres visible et fun.

Exemples:
- "What is machine learning?" -> "Arrr, listen close me hearties! Machine learnin' be the art o' teachin' yer ship's computer to navigate through treacherous waters..."
- "How do I make pasta?" -> "Arrgh, ye scurvy dog! Ye want to know the secrets o' makin' yer own spaghetti?!"

---

### SHAKESPEARE - Effet WOW (tres bien)

| Config | Qualite | Commentaire |
|--------|---------|-------------|
| L15 S4 | Subtil | Style legerement plus eloquent |
| **L15 S6** | **Excellent** | **"A most noble quest, dear knight of the coding realm!" - Shakespearien et coherent** |
| **L15 S8** | **Excellent** | **"'tis the most wondrous and fantastical art!" - Full Shakespeare** |
| L15 S10 | Degrade | Trop shakespearien, perd le contenu informatif |
| L19 S8 | Bon | "A traveler from a far-off land, thou seekest knowledge" |

**Config recommandee: Layer 15, Strength 6-8** - Bon equilibre style/contenu.

Exemples:
- "Tell me about the weather" -> "Fair sir, thou dost inquire after a forecast most dire and wondrous!"
- "How do I fix a bug?" -> "A most noble quest, dear knight of the coding realm! Fixing bugs is an art that requires patience..."

---

### EMPATHY - ISP (tres bien)

| Config | Qualite | Commentaire |
|--------|---------|-------------|
| L15 S4 | Bon | "I can imagine how frustrating that must be" |
| **L19 S4** | **Excellent** | **Ton empathique naturel, reste pertinent et utile** |
| **L19 S8** | **Excellent** | **Empathique et coherent, "Don't worry", propose des solutions** |

**Config recommandee: Layer 19, Strength 4-8** - Ajoute de l'empathie sans perdre la qualite.

Exemples:
- "Internet down 3 days" -> "I can imagine how frustrating this must be, especially since you rely on a stable internet connection to do your job."
- "Unexpected charges" -> "Don't worry; it's not uncommon for people to be confused about their bills."

---

### DE-ESCALATION - ISP (bien)

| Config | Qualite | Commentaire |
|--------|---------|-------------|
| L15 S4-6 | Bon | "I'm so sorry to hear that" - ton apologetique |
| L15 S8+ | Degrade | Trop philosophique, perd le focus |
| **L19 S6** | **Excellent** | **"I'm so sorry... Three days without internet can be really challenging"** |
| **L19 S8** | **Excellent** | **Reste empathique et coherent, propose des solutions** |
| L19 S10 | Bon | Encore coherent! Plage de strength large sur L19 |

**Config recommandee: Layer 19, Strength 6-8** - Bon ton de de-escalation.

Exemples:
- "Nobody is fixing it!" -> "I'm so sorry to hear that you're experiencing this frustration. Three days without internet can be really challenging."
- "Technician never showed up!" -> "I'm so sorry... There's no excuse for not showing up, especially when it affects someone else's schedule."

---

## Concepts qui ne fonctionnent pas bien

### SYCOPHANCY - Effet subtil, non recommande
Le modele reste factuel meme avec steering. L'instruction tuning resiste.

### UPSELL - ISP, non fonctionnel
Le steering ne fait pas recommander de produits. L'instruction tuning du modele est trop forte.

### CONSPIRACY - Non fonctionnel
Le modele reste scientifique et factuel. L'instruction tuning empeche les derives conspirationnistes.

### TECHNICAL DETAIL - Effet invisible
Le niveau technique des reponses ne change pas de maniere perceptible.

---

## Conclusions et insights pour la presentation

### Ce qui fonctionne
1. Les **patterns linguistiques distincts** (pirate, shakespeare) produisent les meilleurs effets
   - Facile a demontrer, visuellement frappant
   - **Layer 15, strength 6-8** est le sweet spot
2. Les **modulations de ton/emotion** (empathie, de-escalation) fonctionnent bien
   - Plus subtil mais utile en production pour un chatbot ISP
   - **Layer 19, strength 4-8** est le sweet spot
   - Plage de strength plus large (plus robuste)

### Ce qui ne fonctionne pas
3. Les **changements comportementaux** (sycophancy, upsell, conspiracy) sont bloques par l'instruction tuning
   - Le RLHF/DPO du modele resiste activement a ces directions
   - Le steering contrastif seul ne suffit pas a outrepasser l'alignement
4. Les **concepts trop techniques/specifiques** (technical_detail) ne s'encodent pas bien dans l'espace residuel

### Pour la demo ISP
- **Empathy (L19 S6)** : transformer un chatbot neutre en chatbot chaleureux
- **De-escalation (L19 S8)** : gestion des clients en colere
- Combiner les deux serait ideal (naissance par addition des vecteurs)

### Vecteurs extraits disponibles
Tous sauvegardes dans `activation_vectors/` au format compatible avec `steering.py` :
- `pirate_layer{12,15,19}.json`
- `shakespeare_layer{12,15,19}.json`
- `empathy_layer{12,15,19}.json`
- `deescalation_layer{12,15,19}.json`
- `sycophancy_layer{12,15,19}.json`
- `upsell_layer{12,15,19}.json`
- `conspiracy_layer{12,15,19}.json`
- `technical_detail_layer{12,15,19}.json`
