# Détecter les Bad Buzz grâce à l'analyse de sentiments

Notre client nous a demandé **de prédire le sentiment associé à un tweet** parlant de sa marque pour détecter les éventuels bad-buzz. L'analyse des sentiments avec deux polarités semblent donc parfaitement adapté à la demande.

---
## Qu'est-ce que l'analyse des sentiments ?

L'analyse des sentiments est un cas particulier de classification qui consiste à identifier la polarité *(**positives** / **neutres** / **négatives** ou autre variante... )* d'un texte donnée.

<img src="medias/sentiment-analysis.jpeg" width="500">

C'est donc une technique de traitement du langage naturel *(**NLP**: natural language processing)* qui utilise un modèle d'apprentissage supervisé pour attribuer un *score de sentiment* pondérés aux *entités*, *sujets*, *thèmes* ou *catégories* d'une phrase ou d'un bloc texte entier, afin d'obtenir un score global pour le texte concerné.

> L'analyse des sentiments peut être utilée pour:
> - **évaluer l'opinion publique**,
> - **réaliser des études de marché**,
> - **identifier les ressentis des clients**,
> - **surveiller la réputation des marques et des produits**.

---
## Préparation des données

Nous avons d'abord analysé le jeu de données pour en tirer des informations utiles :

>### 1. Distribution de la polarité des tweets
> <img src="medias/EDA_target_balance.png"><br>
> le jeu de données est **parfaitement distribué** entre les deux labels.
> ---
>### 2. Distribution de la temporalité des tweets
> <img src="medias/EDA_time_distribution.png"><br>
> tous les tweets ont été postés en **2009**, ce qui nous permet de déterminer une **limite de 140 caractères**.
> ---
>### 3. Distribution des tweets selon leur nombre de caractères
> <img src="medias/EDA_chars_distribution.png"><br>
> certains tweets dépassent la limite des 140 caractères... il a donc fallu chercher pourquoi et agir en conséquence.
> ---
>### 4. Distribution des tweets selon leur nombre de mots
> <img src="medias/EDA_words_distribution.png"><br>
> la moyenne se trouve autour de 13 mots et le maximum à 65 mots. Nous allons donc harmonizer la longueur des séquences avec un padding de 65.

<br>Puis après la suppression des balises HTML et des MENTIONS Twitter ou encore le remplacement des URLs par un tag *(nous aurions dû les supprimer)*, nous avons préparé 8 jeux de données ayant reçu différents pré-processing:

> `RAW`
> - aucun pré-traitement<br>
> <img src="medias/wordcloud_data_clean.png" width="300"><br>

> `PREPROCESS01`
> - nettoyage avec Twitter-preprocessor<br>
> <img src="medias/wordcloud_data_preprocessed_01.png" width="300"><br>

> `PREPROCESS02`
> - nettoyage avec Twitter-preprocessor
> - Tokenization<br>
> <img src="medias/wordcloud_tokens.png" width="300"><br>

> `PREPROCESS03`
> - nettoyage avec Twitter-preprocessor
> - Tokenization
> - Filtrage **avancé**<br>
> <img src="medias/wordcloud_tokens_advanced.png" width="300"><br>

> `PREPROCESS03_simple`
> - nettoyage avec Twitter-preprocessor
> - Tokenization
> - Filtrage **simple**<br>
> <img src="medias/wordcloud_tokens_simple.png" width="300"><br>

> `PREPROCESS04`
> - nettoyage avec Twitter-preprocessor
> - Tokenization
> - Filtrage **avancé**
> - Lemmatization<br>
> <img src="medias/wordcloud2.png" width="300"><br>

> `PREPROCESS04_simple`
> - nettoyage avec Twitter-preprocessor
> - Tokenization
> - Filtrage **simple**
> - Lemmatization<br>
> <img src="medias/wordcloud2simple.png" width="300"><br>

> `PREPROCESS04_nofilter`
> - nettoyage avec Twitter-preprocessor
> - Tokenization
> - AUCUN Filtrage
> - Lemmatization<br>
> <img src="medias/wordcloud2nofilter.png" width="300"><br>


---
## Modèle naïf

Nous avons commencé par la mises en place d'un algorithme naïf *(DummyClassifier)* permettant d'établir une base de référence. Pour se faire, nous avons utilisé tous les samples du pré-processing `RAW` sur lesquels nous avons appliqué un **TF-IDF**.

<img src="medias/Dummy_scores.png">
<img src="medias/Dummy_confusion.png">
<img src="medias/Dummy_ROCAUC.png">

---
## Modèle sur mesure simple

Après avoir établi notre baseline naïve, nous avons testé une **régression logistique** avec grid-search pour les hyper-paramètres, que l'on a entraîné sur les différents pré-traitements préparés.

Les 8 modèles produits avec les 8 pré-traitements, on mis en évidence l'influence négative de certains d'entre eux. On note en particulier l'**influence de la suppression des stop-words** qui semble problématique puisque ce sont les pré-traitements ayant gardé les stop-words qui ont obtenus les meilleurs résultats. On peut supposer que leur suppression peut changer le sens des mots *(`I don't like it` --> `I like`)* et donc perturber l'apprentissage.

Voici les résultats obtenus avec tous les samples du pré-processing `RAW` sur lesquels nous avons appliqué un **TF-IDF**

<img src="medias/LR0_scores.png">
<img src="medias/LR0_confusion.png">
<img src="medias/LR0_ROCAUC.png">

---
## Modèle sur mesure avancé

Après avoir obtenu une idée raisonnable de ce que pouvait nous offrir un modèle « simple », nous avons exploré les possibilités offertes par les réseaux de neurones *(**Neural-Networks**)* et en particulier les réseaux de neurones récurrents *(**RNN**)*.

Pour ce faire, nous avons procédé étape par étape:

>### 1. recherche du **pré-processing** donnant les meilleurs résultats avec un RNN basique.
> Chacun des 8 jeux de données pré-processés ont été traités avec un `Tokenizer` pour transformer les mots en valeurs numériques *(on avait donc des séquences d'index au lieu de séquences de mots)*
>
> Puis ils ont été évalués avec un modèle RNN standard et commun à tous les tests:
>> inputs = keras.Input(shape=(None,), dtype="int64")<br>
>> x = layers.Embedding(input_dim=vocab_size_, output_dim=100, input_length=50, trainable=True)(inputs)<br>
>> x = layers.Bidirectional(layers.LSTM(64))(x)<br>
>> x = layers.Dense(24, activation='relu')(x)<br>
>> predictions = layers.Dense(1, activation='sigmoid', name='predictions')(x)
>
> **Conclusions** : Tout comme avec les tests fait sur la Régression Logistique, les pré-processing ne supprimant PAS les stop-words ont donné les meilleurs résultats.
>
> Mais cette fois ce n'est pas le traitement le plus simple qui s'est imposé, mais le pre-traitement `PREPROCESS04_nofilter` consistant en un filtrage rudimentaire *(celui inclut dans la tokenization, donc pas les stop-words)*, suivi d'une lemmatization.

<br>

>### 2. recherche du **plongement de mots** fonctionnant le mieux avec le pré-processing sélectionnée.
>> Un plongement de mots est **une représentation vectorielle des mots** souvent de basse dimension. Chaque mot est mis en correspondance avec un vecteur et les valeurs des vecteurs sont apprises par entraînement *(certains plongements sont basés sur des réseaux de neurones, d'autres sur les factorisations de matrices)* à l'aide de corpus *(souvent très grands)* qui apportent une dimension contextuelle qui est inclue dans les vecteurs.
>>
>> On cherche donc à obtenir des représentations très proches pour des mots au sens très proches en utilisant et en incluant le contexte dans lequel ils sont présentés. Par exemple les mots `Femme` et `Homme` devraient être encodés de façon assez similaire de sorte que si l'on fait `Roi` - `Homme` + `Femme`, on devrait obtenir le vecteur correspondant au mot `Reine`. L'un des gros avantages de cette approche est d'aider à distinguer des mots qui s'écrivent de la même façon et qui n'ont pourtant pas du tout le même sens.
>
> Après avoir déterminé le pré-traitement le plus adapté à notre projet, nous avons donc entrepris de trouver un plongement adapté.
>
> Une dizaine de configurations ont été testées;
> - des modèles **sans embeddings**, donc des modèles `bag-of-words` ou `bag-of-N-gram`,
> - des modèles avec un **embedding keras simple**,
> - des modèles avec des **embeddings pre-entrainés** comme `Word2Vec-300d`, `FastText-300d`, `GloVe-100d`, `GloVe-Twitter-25d`, `GloVe-Twitter-100d` ou `GloVe-Twitter-200d`.
>
> **Conclusions** : certaines configurations sans embeddings *(en particuliers les bags of bigrams en one-hot-encoding, en count ou en tf-idf)* fonctionnent assez bien, mais sont confronté à de potentiels problèmes en lien avec la taille du vocabulaire *(sauf si l'on restreint sa taille et que donc on prend le risque de mal représenter le corpus)*.
>
> Les modèles avec un embedding pré-entrainé donnent d'excellents résultats pour certains et des résultats beaucoup plus mitigés pour d'autres. Par exemple les modèles les plus connus, mais aussi les plus génériques comme le `Word2Vec300_GoogleNews` ou le `FastText300` ont donnés des résultats très honorables sans pour autant surpasser notre meilleur modèle bag of bigrams *(un modèle en one-hot-encoding)*. De même, le `Glove-100d` n'a pas brillé sur notre jeu de données, mais les versions de GloVe spécialement entraînées sur des corpus Twitter ont donné d'excellents résultats.
>
> C'est d'ailleurs l'embedding `GloveTwitter 200d` *(qui était à égalité avec le 100d)* qui a retenu notre attention pour la suite.

<br>

>### 3. recherche de **l'architecture** la plus efficace avec les choix précédents.
>
> Durant cette étape, 11 architectures basées sur des couches `SimpleRNN`, `LSTM` *(Long Short Term Memory)* ou encore `GRU` *(Gated Recurrent Unit)* ont été comparées. On a essayé des variantes en uni ou bidirectionnel ou encore des variantes avec un plus ou moins grand nombre d'unitées.
>
>> Les réseaux neuronaux récurrents *(réseaux neuronaux dont au moins un neurone tourne en boucle vers sa propre couche ou une couche précédente)* tels que les *SimpleRNN*, les *LSTM* ou les *GRU*, sont des réseaux de neurones crées pour traiter des données de manière séquentielle en essayant de faire persister l'information.
>
> **Conclusions** : En regardant les résultats, plusieurs tendances se dessinent;
> - les architectures avec des couches *LSTM* et *GRU* *(qui se valent dans l'ensemble)* surpassent les architectures plus simples,
> - les architectures bi-directionnelles surpassent les uni-directionnelles,
> - plus on a d’unités sur les couches et plus ça semble bénéfique.
>
> Et c'est `l'architecture la plus complexe qui nous a donné les meilleurs résultats`, mais les différences entre toutes les architectures *LSTM* et *GRU* essayées sont finalement assez faibles *(moins de 2% d'écart)* et ces tendances pourraient fort bien être tout autre avec un jeu de données différent.

Voici pour comparaison, les résultats du meilleurs modèle RNN obtenu:

<img src="medias/RNN402_scores.png">
<img src="medias/RNN402_confusion.png">
<img src="medias/RNN402_ROCAUC.png">

---
## Modèle avancé BERT

Enfin, nous avons exploré les possibilités offertes par les transformers et en particulier les modèles BERT *(Bidirectional Encoder Representations from Transformers)*.

> BERT est une technique d'apprentissage automatique basée sur les transformers développée par Google et utilisé pour le traitement du langage naturel *(NLP)*.
>
> Le Transformer est un modèle de Deep Learning *(donc un réseau de neurones)* de type *seq2seq* *(un modèle qui prend en entrée une séquence et renvoie une séquence en sortie)* qui a la particularité de n’utiliser que le mécanisme d’attention et aucun réseau récurrent ou convolutionnel. L’idée sous jacente est de conserver l’interdépendance des mots d’une séquence en n’utilisant pas de réseau récurrent mais seulement le mécanisme d’attention qui est au centre de son architecture.

Pour ce faire, nous avons essayé:

>### 1. BERT sans fine-tuning
>
> ces modèles pré-entrainés sont rapide à mettre en place, mais à moins de choisir une variante correpondant vraiment à notre jeu de données, les performances restent en dessous de celles obtenus avec notre RNN. Par ailleurs, bien que l'on puisse visiblement s'en passer, un **GPU semble nécessaire** pour obtenir des temps d'inférence décents...

>### 2. BERT avec fine-tuning
>
> les modèles avec fine-tuning produisent `des performances très intéressantes`; le modèle entrainé pour cet essai obtient une augmentation de plus de 3% d'accuracy par rapport au meilleurs modèle RNN avec seulement 100 000 samples.
> 
> Mais ce gain se paie le prix fort avec **un temps d'entrainement particulièrement long *(même avec 1 ou plusieurs GPU)***. Nous pourrions donc certainement améliorer le modèle en lui fournissant davantages d'exemples, mais il faudra en payer le prix en temps *(et donc en argent)*.

Voici pour comparaison, les résultats du meilleurs modèle BERT obtenu:

<img src="medias/TRFT1_scores.png">
<img src="medias/TRFT1_confusion.png">
<img src="medias/TRFT1_ROCAUC.png">

---
## Conclusions


<table>
    <tr>
        <th>Method</th>
        <th>ROC AUC</th>
        <th>Accuracy</th>
        <th>Training-time</th>
        <th>Inference-time</th>
        <th>Total samples</th>
    </tr>
    <tr>
        <th>DummyClassifier (RAW + TF-IDF)</th>
        <td>0.501126</td>
        <td>0.501144</td>
        <td>2.180013</td>
        <td>0.026919</td>
        <td>1452791</td>
    </tr>
    <tr>
        <th>LogisticRegression (RAW + TF-IDF)</th>
        <td>0.878261</td>
        <td>0.797955</td>
        <td>1104.552671</td>
        <td>0.031679</td>
        <td>1452791</td>
    </tr>
    <tr>
        <th>RNN Archi-402 (lemmas_not_filtered + TextVectorization[int])</th>
        <td>0.912817</td>
        <td>0.831697</td>
        <td>829.031158</td>
        <td>40.984504</td>
        <td>1452791</td>
    </tr>
    <tr>
        <th>Transformers Fine-Tuning_1 (RAW + AutoTokenizer)</th>
        <td>0.937993</td>
        <td>0.864</td>
        <td>7363.0</td>
        <td>40.7827</td>
        <td>100000</td>
    </tr>
</table>

On voit clairement une évolution des métriques de nos modèles, et dans l'absolu le modèle *roberta-base* avec fine-tuning nous donne les meilleurs résultats et de loin, mais il demande de grosses ressources pour être bien entrainé et correctement déployé.

Le modèle que nous avons choisi de déployer pour le moment est donc le meilleurs modèle RNN que nous avons obtenu, qui malgré des performances légèrement moins intéressantes à l'avantage d'être entrainable dans un laps de temps beaucoup plus raisonnable et de ne pas avoir besoin de GPU pour fonctionner *(il est d'ailleurs déployé sans GPU sur Heroku et offre des temps de réponse tout à fait acceptables)*.