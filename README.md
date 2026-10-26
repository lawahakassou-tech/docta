# docta

PUZZLE.CU - Recherche de clés privées Bitcoin (secp256k1) : tous les GPU + tous les CPU
=========================================================================================

CE QUE FAIT LE PROGRAMME
------------------------
Il teste des clés privées consécutives dans une plage donnée (bits ou début:fin) et compare le résultat
(clé publique compressée, ou adresse 1... / bc1q...) à une ou plusieurs cibles (fichier .txt, filtre .bloom).
C'est le programme docta_v2.cpp (CPU) auquel on a ajouté :

  * un moteur CUDA (arithmétique 32 bits, inversion par lots, SHA-256 + RIPEMD-160 sur GPU) ;
  * l'utilisation de TOUS les GPU détectés (Kaggle : 2 x T4 ; Colab : T4 / L4 / A100 / H100 ...) ;
  * l'utilisation de TOUS les cœurs CPU en même temps que les GPU (les cœurs sont lus dans l'affinité et le
    quota cgroup du conteneur, donc pas de sur-souscription sous Colab/Kaggle) ;
  * un partage des blocs commun CPU+GPU : chaque bloc n'est testé qu'une seule fois, quel que soit le mode
    (-r, -s, -sobol, -pi, -e, -ln, -m, et -v R) ;
  * un point de reprise (checkpoint) : on peut arrêter / relancer (fin de session Kaggle/Colab) sans rien perdre ;
  * un affichage adapté aux notebooks (une ligne d'état toutes les 15 s quand la sortie n'est pas un terminal) ;
  * des auto-tests au démarrage qui vérifient l'arithmétique et le noyau sur CHAQUE GPU avant de chercher.

Tous les candidats trouvés par un GPU sont revérifiés côté CPU (k*G recalculé en arithmétique affine, comparaison
exacte avec la liste des cibles) avant d'être enregistrés. La clé est écrite dans deux fichiers
(cle_trouvee.txt et cle_trouvee_copie.txt), relus après écriture, et affichée dans la sortie du notebook.


A SAVOIR (HONNÊTETÉ SUR LES ORDRES DE GRANDEUR)
------------------------------------------------
Une clé privée fait 256 bits : fouiller tout l'espace est impossible, avec n'importe quel matériel. Ce programme n'a de
sens que sur une plage RÉDUITE et connue (ex. les "puzzles" Bitcoin dont on sait que la clé est dans [2^(n-1), 2^n-1],
ou une plage que l'on a de bonnes raisons de croire). Ordres de grandeur (à mesurer avec --time sur votre session) :
  - un cœur CPU : ~1 à 3 millions de clés/s ; un T4 : de l'ordre de quelques centaines de millions de clés/s
    (moins en mode "address", car le hachage coûte plus cher que la courbe) ;
  - 2^40 clés ~ de l'ordre de l'heure sur 1 T4 ; 2^50 ~ plusieurs semaines ; 2^66 et plus : hors de portée de Kaggle/Colab.
Les sessions sont limitées (Kaggle ~9-12 h par session et quota GPU hebdomadaire ; Colab gratuit : déconnexion
après inactivité / quelques heures) : utilisez la reprise (voir plus bas) et --out vers un dossier persistant.


FICHIERS
--------
  puzzle.cu   le programme (un seul fichier : CPU + GPU)
  readme.txt  ce fichier


1. COMPILATION
--------------
Il faut nvcc (déjà présent sur Kaggle et Colab : /usr/local/cuda/bin/nvcc).

  Cas général, un seul type de GPU dans la machine (recommandé : le plus rapide à compiler) :
      nvcc -O3 -std=c++17 -arch=native -Xcompiler -O3,-march=native,-pthread -lpthread -o puzzle puzzle.cu

  Si "-arch=native" n'est pas reconnu (nvcc ancien) : choisir l'architecture à la main
      T4 / RTX 20xx -> sm_75     P100 -> sm_60     V100 -> sm_70     A100 -> sm_80
      RTX 30xx -> sm_86          L4 / RTX 40xx -> sm_89            H100 -> sm_90
      nvcc -O3 -std=c++17 -gencode arch=compute_75,code=sm_75 -Xcompiler -O3,-march=native,-pthread -lpthread -o puzzle puzzle.cu

  Version CPU seulement (aucune dépendance à CUDA, pour tester ou machine sans GPU) :
      g++ -x c++ -O3 -march=native -std=c++17 -pthread -o puzzle_cpu puzzle.cu

  La compilation prend de 1 à 4 minutes (le noyau GPU est volumineux) : c'est normal.
  Les avertissements "ignoring #pragma ..." de g++ sont sans conséquence.


2. KAGGLE (Notebook > Settings > Accelerator : "GPU T4 x2" ou "GPU P100")
-------------------------------------------------------------------------
Importer puzzle.cu comme "Dataset" ou le coller avec %%writefile. Cellules :

  Cellule 1 - vérifier le matériel
      !nvidia-smi -L
      !nvcc --version

  Cellule 2 - écrire le fichier (ou copier depuis le dataset : !cp /kaggle/input/<nom>/puzzle.cu .)
      %%writefile puzzle.cu
      ... coller le contenu de puzzle.cu ...

  Cellule 3 - compiler pour le(s) GPU présent(s) (détection automatique de l'architecture)
      import subprocess
      cc = subprocess.run(["nvidia-smi","--query-gpu=compute_cap","--format=csv,noheader"],
                          capture_output=True,text=True).stdout.split()
      archs = sorted(set(c.replace(".","") for c in cc))
      gen = " ".join(f"-gencode arch=compute_{a},code=sm_{a}" for a in archs)
      cmd = f"nvcc -O3 -std=c++17 {gen} -Xcompiler -O3,-march=native,-pthread -lpthread -o puzzle puzzle.cu"
      print(cmd); r = subprocess.run(cmd,shell=True,capture_output=True,text=True); print(r.stderr[-2000:])

  Cellule 4 - voir ce qui est détecté
      !./puzzle --info

  Cellule 5 - test rapide (génère une adresse de test et la retrouve ; clé 0x2a3c5f = 22 bits)
      !./puzzle --genaddr 2a3c5f
      # recopier l'adresse "1..." affichée :
      !./puzzle 1XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX 22 address --out /kaggle/working --status 2

  Cellule 6 - la vraie recherche, en tâche de fond (la cellule n'est pas bloquée) et reprise automatique
      !nohup ./puzzle <adresse_ou_clé_publique> <bits|début:fin> address -sobol \
             --out /kaggle/working --time 11h > run.log 2>&1 &
      # suivre :
      !tail -n 15 run.log

  Les GPU (2 x T4) et les cœurs CPU (4 vCPU) sont tous utilisés automatiquement.
  Récupération des résultats : /kaggle/working/cle_trouvee.txt (visible dans l'onglet "Output" après
  "Save Version" / "Save & Run All"). La reprise s'appuie sur /kaggle/working/puzzle_<empreinte>.ckpt.
  Astuce : un notebook lancé avec "Save & Run All (Commit)" continue en arrière-plan même onglet fermé.


3. GOOGLE COLAB (Exécution > Modifier le type d'exécution > GPU : T4 / L4 / A100 ...)
--------------------------------------------------------------------------------------
Mêmes cellules que Kaggle (Colab donne un seul GPU, parfois 2 sur certains plans). Pour ne rien perdre à la
déconnexion, monter Google Drive et y écrire résultats + point de reprise :

      from google.colab import drive
      drive.mount('/content/drive')
      !mkdir -p /content/drive/MyDrive/puzzle
      !nohup ./puzzle <cible> <bits|début:fin> address -sobol \
             --out /content/drive/MyDrive/puzzle --time 11h > run.log 2>&1 &
      !tail -n 15 run.log

Après une déconnexion : ré-ouvrir une session, recompiler (cellule 3), relancer EXACTEMENT la même commande :
le programme détecte le fichier puzzle_<empreinte>.ckpt dans --out et affiche "Reprise : ...".
(Le nom du point de reprise dépend de la plage et des cibles, pas du matériel : on peut reprendre sur Colab une
recherche commencée sur Kaggle en copiant le fichier .ckpt.)


4. UTILISATION
--------------
  ./puzzle <cible> <bits | début:fin> [address] [mode] [options]

  <cible>        clé publique compressée (02../03.., 66 hex), adresse (1... ou bc1q...), fichier .txt (une cible par
                 ligne ; construit et sauvegarde un .bloom) ou fichier .bloom.
  <bits>         plage [2^(bits-1), 2^bits - 1] ; ou plage hexadécimale libre, ex. 2000000:3ffffff
  address        à ajouter quand les cibles sont des adresses
  mode           -r aléatoire (défaut)  -s séquentiel  -sobol  -pi  -e  -ln  -m Monte Carlo
  -v R           voisinage : chaque clé de la suite est le centre d'une fenêtre [k-R, k+R] testée en entier
                 (R >= 512 conseillé)
  -t N           cœurs CPU utilisés (défaut : tous ; 0 = GPU seuls)
  --gpus L       all (défaut) | none | indices, ex. 0,1      (--no-gpu = none)
  --out DIR      dossier des résultats et du point de reprise
  --ckpt FILE    fichier de reprise explicite ; --no-ckpt le désactive ; --fresh ignore un point de reprise existant
  --ckpt-every S intervalle de sauvegarde (défaut 120 s)
  --time T       arrêt propre après T (3600, 90m, 11h30m ...) : point de reprise écrit, puis on peut relancer
  --status S     intervalle des lignes d'état hors terminal (défaut 15 s)
  --skip-test    saute les auto-tests (reprise plus rapide ; déconseillé au premier lancement)
  --info         liste les GPU et le nombre de cœurs
  --selftest     lance uniquement les auto-tests (CPU + GPU)
  --genpub K | --genaddr K   affiche la clé publique / l'adresse de la clé privée K (hex), pour tester

  Exemples
      ./puzzle 03a7...66hex... 40                          # clé publique, 40 bits, aléatoire, tout le matériel
      ./puzzle 1Abc...  2000000000:3fffffffff address -s   # adresse, plage libre, séquentiel
      ./puzzle cibles.txt 45 address -sobol -v 10000       # plusieurs adresses, voisinage
      ./puzzle <cible> 38 --gpus 0 -t 2                    # un seul GPU + 2 cœurs CPU
      ./puzzle <cible> 38 --time 90m --out /kaggle/working # s'arrêter proprement avant la fin de session

  Lecture de la sortie (hors terminal, donc dans un notebook) :
      [ 12.34%] 812.5 Mk/s (GPU 805.1 | CPU 7.40) | écoulé 00:05:12 | trouvées 0/1
  En cas de découverte : "*** CLÉ TROUVÉE ***" est affiché aussitôt et la clé est écrite dans cle_trouvee.txt.
  Code de retour : 0 = toutes les cibles trouvées ; 1 = recherche terminée sans tout trouver ; 130 = interrompu.

  Remarque sur les modes avec beaucoup de GPU : chaque thread GPU travaille sur son propre bloc ; les modes -s et -sobol
  parcourent donc "en parallèle" les premiers blocs de la suite (même couverture, ordre approximatif).


5. POINT DE REPRISE
-------------------
Toutes les 120 s (et à l'arrêt, Ctrl-C/SIGTERM/--time) le programme écrit puzzle_<empreinte>.ckpt dans --out :
liste des blocs terminés + position des blocs en cours + nombre de clés déjà testées. Le découpage en blocs ne dépend
que de la plage (pas du matériel), donc une reprise est possible sur une autre machine / un autre nombre de GPU et
même avec un autre mode. Un fichier incompatible (autre plage, autres cibles) est ignoré avec un message, jamais
utilisé. Un bloc n'est marqué "terminé" que lorsqu'il a été entièrement parcouru.
Si une session est tuée brutalement, on perd au plus ~2 minutes de travail (le dernier intervalle).


6. DÉPANNAGE
------------
  "aucun GPU CUDA détecté"        activer l'accélérateur GPU du notebook ; relancer ./puzzle --info.
  "nvcc: command not found"       essayer /usr/local/cuda/bin/nvcc (ou export PATH=$PATH:/usr/local/cuda/bin).
  "Unsupported gpu architecture"  utiliser -gencode arch=compute_XX,code=sm_XX avec la valeur du GPU (voir plus haut).
  "AUTO-TEST ÉCHEC ... (GPU n)"   le noyau ne donne pas les bons résultats sur ce GPU : le programme refuse de chercher
                                  (une recherche fausse serait silencieusement perdue). Envoyer le message exact.
  "GPU n arrêté : ..."            erreur CUDA en cours de route (mémoire, pilote) : les autres GPU et le CPU continuent ;
                                  les blocs en cours de ce GPU sont conservés dans le point de reprise.
  Compilation très lente          normal avec plusieurs -gencode ; n'en mettre qu'un (celui de votre GPU).
  Moins de Mk/s que prévu         les cœurs CPU partagent la machine avec le système ; essayer -t 0 (GPU seuls) pour
                                  comparer. Le mode "address" est plus lent que "clé publique" (hachage).
  Cibles très nombreuses          le filtre de Bloom est chargé dans la mémoire de chaque GPU (64 octets par 16 cibles).


7. NOTES TECHNIQUES
-------------------
  - Corps Fp : 8 mots de 32 bits, réduction p = 2^256 - 2^32 - 977, carré dédié, inverse par chaîne d'additions.
  - Un thread GPU traite un lot de 256 clés consécutives (centre C, clés C+-i) avec UNE inversion modulaire par lot
    (astuce de Montgomery), comme le moteur CPU (qui traite 1024 clés par lot).
  - Chaque lancement de noyau dure ~0,25 s (réglage automatique) : réactif à Ctrl-C, sans perte de rendement.
  - Candidats : filtre de Bloom par blocs de 64 octets testé sur GPU ; les passages sont renvoyés à l'hôte qui
    vérifie (liste triée exacte, puis recalcul indépendant de k*G).
  - Validation faite pendant le développement : le code "périphérique" est le même code C compilé pour l'hôte ; il a
    été comparé au moteur CPU de référence (milliers d'opérations aléatoires, cas limites, frontières de lots, plages
    quelconques, voisinage, reprise) et exécuté en émulation. Les auto-tests refont l'essentiel de ces contrôles
    sur chaque vrai GPU au démarrage. Les vitesses ci-dessus sont des ordres de grandeur, à confirmer sur votre session.
