"""
Evaluation question sets for the RAG pipeline.

Split by difficulty so rag_eval.py can pick which one to run:
- simple:   the whole answer is one clear sentence, no ambiguity, no table.
- medium:   the answer sits inside a table (several similar rows/columns),
            so the model has to pick out the right cell, not just the right sentence.
- advanced: needs either a synthesis across 2 documents, a multi-step
            reasoning chain, or a comparison across several values at once.
"""

EVALUATION_SET_simple = [

    {
        "question": "Quelle est la durée du préavis de démission pour un cadre chez TechNova Solutions ?",
        "expected_keywords": ["3 mois"],
        "expected_source": "01_politique_rh.pdf",
        "expected_page": 2,
    },

    {
        "question": "Quelle est la garantie matérielle standard de la passerelle NovaEdge G3 ?",
        "expected_keywords": ["3 ans", "5 ans"],
        "expected_source": "09_guide_installation_novaedge.pdf",
        "expected_page": 1,
    },

    {
        "question": "Combien de temps la passerelle NovaEdge G3 peut-elle stocker les mesures en cas de coupure réseau ?",
        "expected_keywords": ["72 heures"],
        "expected_source": "07_documentation_produit.pdf",
        "expected_page": 1,
    },

]

EVALUATION_SET_medium = [

    {
        "question": "Quel est le crédit de service accordé si la disponibilité constatée est comprise entre 99,0 % et moins de 99,5 % ?",
        "expected_keywords": ["10 %"],
        "expected_source": "04_contrat_prestation_services.pdf",
        "expected_page": 2,
    },

    {
        "question": "Combien de règles d'alerte actives sont incluses dans le plan Advanced de NovaCloud ?",
        "expected_keywords": ["500"],
        "expected_source": "07_documentation_produit.pdf",
        "expected_page": 0,
    },

    {
        "question": "Quel est le délai de prise en charge pour un incident de sécurité classé P2 - Majeur ?",
        "expected_keywords": ["1 heure ouvrée"],
        "expected_source": "02_procedures_securite.pdf",
        "expected_page": 1,
    },

]

EVALUATION_SET_advanced = [

    {
        "question": "En cas de violation de données personnelles affectant un client, quel délai le Prestataire doit-il respecter pour notifier le client, et quel délai le DPO doit-il respecter pour notifier la CNIL ?",
        "expected_keywords": ["48 heures", "72 heures"],
        "expected_source": "04_contrat_prestation_services.pdf",
        "expected_page": 1,
    },

    {
        "question": "Si une alerte P1 n'est pas acquittée par l'ingénieur de niveau N1 dans le délai imparti, que se passe-t-il et sous quel délai supplémentaire ?",
        "expected_keywords": ["15 minutes", "N2"],
        "expected_source": "10_runbook_exploitation_astreinte.pdf",
        "expected_page": 0,
    },

    {
        "question": "Un client du plan Standard passe au plan Enterprise : que deviennent les limites de règles d'alerte et de capteurs ?",
        "expected_keywords": ["5 000", "Illimité"],
        "expected_source": "07_documentation_produit.pdf",
        "expected_page": 0,
    },

]
